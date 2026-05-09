"""
Bot 1 – News Sentiment Bot  (cost-optimised)
Model  : Haiku — ONLY when urgent keywords detected (2-3×/day)
Normal : keyword scoring only — zero Claude cost

Operating modes:
  Normal scan  (every 15 min) — fetch articles, keyword score, publish.  No Claude.
  RSS watchdog (every  5 min) — fetch Yahoo RSS headlines, keyword check only.
                                 If urgent term found → trigger Urgent scan.
  Urgent scan  (on trigger)   — single Haiku call for that article batch.
                                 Telegram alert published immediately.
  9am catalyst scan (once/day)— Haiku call on overnight/pre-market news.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
import aiohttp
import pytz

import anthropic

from config import Models, RedisConfig, UniverseConfig, AnthropicConfig, AlertConfig
from shared.base_bot import BaseBot
from shared.llm_router import LLMRouter


NORMAL_SCAN_INTERVAL = 900    # 15 min — keyword only, no Claude
RSS_CHECK_INTERVAL   = 300    # 5 min  — headlines only, no Claude
URGENT_COOLDOWN_SECS = 600    # min seconds between urgent Claude calls
MARKET_TZ            = pytz.timezone("America/New_York")

# ── Breaking news keyword groups ───────────────────────────────────────────────
# Matched as whole words (case-insensitive) against the raw headline.
# Any match → urgent Haiku call + Telegram alert.
URGENT_KEYWORDS: dict[str, set[str]] = {
    "earnings":   {"earnings", "beat", "miss", "eps", "guidance", "outlook"},
    "fda":        {"fda", "approved", "approval", "rejected", "rejection", "clearance"},
    "m&a":        {"merger", "acquisition", "buyout", "takeover", "acquires"},
    "price_move": {"crash", "plunge", "surge", "halt", "halted", "circuit breaker"},
    "fed":        {"fed", "powell", "fomc", "rate hike", "rate cut", "inflation"},
    "macro":      {"war", "attack", "crisis", "emergency", "sanctions", "default"},
    "legal":      {"bankrupt", "bankruptcy", "fraud", "sec charges", "indictment"},
}
# Flat set for fast O(1) per-word checks
_ALL_URGENT: frozenset[str] = frozenset(
    w for words in URGENT_KEYWORDS.values() for w in words
)


class NewsSentimentBot(BaseBot):
    """
    Bot 1 – News Sentiment Bot
    Two parallel loops: 15-min normal scan (keyword only) + 5-min RSS watchdog.
    Claude Haiku is called only when urgent keywords are detected.
    """

    BOT_ID = 1
    NAME   = "News Sentiment Bot"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.HAIKU)
        self.client   = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self.news_api = AlertConfig.NEWS_API_KEY
        self._router  = LLMRouter(self.client)

        self._seen_ids: set[str]           = set()    # headline dedup
        self._seen_urgent: set[str]        = set()    # urgent headline dedup
        self._last_urgent_call: float      = 0.0      # monotonic ts of last Haiku call
        self._catalyst_scan_done: bool     = False
        self._urgent_calls_today: int      = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        self.log(
            f"News Sentiment Bot starting | normal={NORMAL_SCAN_INTERVAL//60}min "
            f"rss_check={RSS_CHECK_INTERVAL//60}min | Claude only on urgent keywords"
        )
        asyncio.create_task(self._morning_catalyst_task())

    async def run(self):
        # Both loops run concurrently; neither blocks the other
        await asyncio.gather(
            self._normal_scan_loop(),
            self._rss_watchdog_loop(),
        )

    async def cleanup(self):
        self.log(
            f"News Sentiment Bot stopped | urgent Claude calls today: {self._urgent_calls_today}"
        )

    # ── Normal scan loop — keyword only, no Claude ────────────────────────────

    async def _normal_scan_loop(self):
        """Every 15 min: fetch articles → keyword score → publish.  Zero Claude cost."""
        while self.running:
            try:
                await self._normal_cycle()
            except Exception as e:
                self.log(f"Normal scan error: {e}", "error")
            await asyncio.sleep(NORMAL_SCAN_INTERVAL)

    async def _normal_cycle(self):
        symbols  = list(dict.fromkeys(UniverseConfig.WATCHLIST))
        articles = await self._fetch_alpaca_news(symbols, hours=4)
        source   = "alpaca"
        if not articles and self.news_api:
            articles = await self._fetch_newsapi(symbols, hours=4)
            source   = "newsapi"
        if not articles:
            articles = await self._fetch_yahoo_rss()
            source   = "yahoo_rss"

        self.log(f"Normal scan: {len(articles)} articles from {source}")

        new_articles = self._dedup(articles)
        if not new_articles:
            self.log("Normal scan: no new articles")
            return

        # Keyword score only — no Claude
        scored = self._keyword_score_articles(new_articles)
        await self._publish_scored(scored, mode="normal")

    # ── RSS watchdog — 5-min urgent keyword check, no Claude ──────────────────

    async def _rss_watchdog_loop(self):
        """
        Every 5 min: fetch Yahoo RSS headlines only (free, no API key).
        If an urgent keyword is found and cooldown has passed → urgent scan.
        """
        while self.running:
            await asyncio.sleep(RSS_CHECK_INTERVAL)
            try:
                await self._rss_check()
            except Exception as e:
                self.log(f"RSS watchdog error: {e}", "error")

    async def _rss_check(self):
        articles = await self._fetch_yahoo_rss()
        if not articles:
            return

        for article in articles:
            headline = article.get("title", "")
            uid      = article.get("_id", "")

            if uid in self._seen_urgent:
                continue

            triggered, kw_group, matched_word = self._check_urgent_keywords(headline)
            if not triggered:
                continue

            self._seen_urgent.add(uid)
            # Trim seen set
            if len(self._seen_urgent) > 500:
                self._seen_urgent = set(list(self._seen_urgent)[-250:])

            self.log(
                f"URGENT keyword '{matched_word}' ({kw_group}) in: \"{headline[:80]}\""
            )
            await self._urgent_scan(article, kw_group, matched_word)
            # One urgent scan per RSS pass — avoid flooding
            return

    # ── Urgent scan — Haiku + Telegram alert ─────────────────────────────────

    async def _urgent_scan(self, trigger_article: dict, kw_group: str, matched_word: str):
        """
        Called when urgent keyword detected.
        Fetches a small article batch, calls Haiku once, publishes alert.
        Respects a 10-minute cooldown to avoid repeat calls on the same event.
        """
        import time
        now = time.monotonic()
        if now - self._last_urgent_call < URGENT_COOLDOWN_SECS:
            remaining = int(URGENT_COOLDOWN_SECS - (now - self._last_urgent_call))
            self.log(
                f"Urgent scan skipped: cooldown active ({remaining}s remaining)", "warning"
            )
            return

        self._last_urgent_call   = now
        self._urgent_calls_today += 1
        self.log(
            f"URGENT SCAN #{self._urgent_calls_today} triggered by '{matched_word}' "
            f"({kw_group}) — calling Haiku"
        )

        # Fetch a small batch of fresh articles around this event
        symbols  = list(dict.fromkeys(UniverseConfig.WATCHLIST))
        articles = await self._fetch_alpaca_news(symbols, hours=1)
        if not articles:
            articles = await self._fetch_yahoo_rss()
        # Ensure the trigger article is in the batch
        if trigger_article not in articles:
            articles = [trigger_article] + articles[:9]

        scored = await self._haiku_score_articles(articles[:10])
        if not scored:
            scored = self._keyword_score_articles([trigger_article])

        # Publish scored events
        await self._publish_scored(scored, mode="urgent")

        # Find the most impactful scored result (by score desc)
        best = max(scored, key=lambda s: abs(float(s.get("score") or 0.5)), default=None)
        if best and best.get("sentiment") != "neutral":
            await self._send_breaking_alert(best, matched_word)

    # ── Haiku scoring (urgent only) ────────────────────────────────────────────

    async def _haiku_score_articles(self, articles: list[dict]) -> list[dict]:
        """Call Claude Haiku to score a small batch.  Used only in urgent mode."""
        if not articles:
            return []

        items  = [{"id": i, "t": a.get("title", "")[:90]} for i, a in enumerate(articles)]
        prompt = (
            "Sentiment score each headline. Identify tickers affected.\n"
            f"{LLMRouter.j(items)}\n"
            "JSON:{\"results\":[{\"id\":0,\"sentiment\":\"bullish\","
            "\"score\":0.85,\"symbols\":[],\"catalyst\":\"\"}]}"
        )
        raw = await self._router.call(
            [{"role": "user", "content": prompt}],
            prefer="haiku",
            max_tokens=600,
            timeout=15.0,
        )
        if not raw:
            self.log("Haiku urgent score returned empty — keyword fallback", "warning")
            return self._keyword_score_articles(articles)

        try:
            cleaned = self._extract_json_block(
                re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
            )
            parsed    = json.loads(cleaned)
            if isinstance(parsed, list):
                parsed = {"results": parsed}
            results   = {r["id"]: r for r in parsed.get("results", []) if "id" in r}
        except Exception as e:
            self.log(f"Haiku urgent parse error: {e} — keyword fallback", "warning")
            return self._keyword_score_articles(articles)

        scored = []
        for i, article in enumerate(articles):
            r = results.get(i, {})
            if not r:
                r = self._keyword_score_one(article.get("title", ""))
            scored.append({
                "title":     article.get("title", ""),
                "url":       article.get("url", ""),
                "published": article.get("publishedAt", ""),
                "sentiment": r.get("sentiment", "neutral"),
                "score":     r.get("score", 0.6),
                "symbols":   r.get("symbols", article.get("_symbols", [])),
                "catalyst":  r.get("catalyst", ""),
            })
        return scored

    # ── Telegram breaking-news alert ──────────────────────────────────────────

    async def _send_breaking_alert(self, article: dict, trigger_word: str):
        """Publish a high-priority Telegram alert for breaking news."""
        sentiment = article.get("sentiment", "neutral")
        symbols   = article.get("symbols", [])
        title     = article.get("title", "")
        catalyst  = article.get("catalyst", trigger_word)

        direction = "BULLISH 📈" if sentiment == "bullish" else "BEARISH 📉" if sentiment == "bearish" else "NEUTRAL"
        sym_str   = " ".join(symbols[:6]) if symbols else "Market-wide"

        message = (
            f"🚨 <b>BREAKING NEWS</b>\n\n"
            f"{title}\n\n"
            f"Impact: {direction}\n"
            f"Affected: {sym_str}\n"
            f"Catalyst: {catalyst}\n"
            f"Action: Consider adjusting positions"
        )

        await self.publish(RedisConfig.CHANNEL_ALERTS, {
            "type":     "telegram_alert",
            "priority": "urgent",
            "message":  message,
            "source":   "breaking_news",
            "sentiment": sentiment,
            "symbols":  symbols,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.log(
            f"Breaking alert sent | {sentiment} | affected={sym_str} | "
            f"\"{title[:60]}\""
        )

    # ── Urgent keyword detection ───────────────────────────────────────────────

    @staticmethod
    def _check_urgent_keywords(headline: str) -> tuple[bool, str, str]:
        """
        Returns (triggered, kw_group, matched_word).
        Matches whole words and two-word phrases; case-insensitive.
        """
        lower = headline.lower()
        # Two-word phrases checked first (more specific, fewer false positives)
        phrases = {
            "rate hike", "rate cut", "sec charges", "circuit breaker",
            "trading halt", "rate pause", "federal reserve",
        }
        for phrase in phrases:
            if phrase in lower:
                group = next(
                    (g for g, kws in URGENT_KEYWORDS.items() if phrase in kws),
                    "macro"
                )
                return True, group, phrase

        # Single-word check
        words = set(re.findall(r"\b\w+\b", lower))
        for word in words & _ALL_URGENT:
            group = next(
                (g for g, kws in URGENT_KEYWORDS.items() if word in kws), "general"
            )
            return True, group, word

        return False, "", ""

    # ── 9am Catalyst scan (once/day) — unchanged ──────────────────────────────

    async def _morning_catalyst_task(self):
        """At 9:00am EST, run a targeted catalyst scan once per day."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now_et).total_seconds())
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            self._catalyst_scan_done = False
            self._urgent_calls_today = 0   # reset daily counter
            await self._run_catalyst_scan()
            self._catalyst_scan_done = True

    async def _run_catalyst_scan(self):
        self.log("9am catalyst scan starting…")
        symbols  = list(dict.fromkeys(
            UniverseConfig.WATCHLIST + getattr(UniverseConfig, "SCAN_UNIVERSE", [])
        ))
        articles = await self._fetch_alpaca_news(symbols, hours=12)
        if not articles and self.news_api:
            articles = await self._fetch_newsapi(symbols, hours=12)
        if not articles:
            articles = await self._fetch_yahoo_rss()
        if not articles:
            self.log("Catalyst scan: no articles found")
            return

        catalysts = await self._identify_catalysts(articles)
        if not catalysts:
            self.log("Catalyst scan: no high-priority catalysts found")
            return

        for cat in catalysts:
            await self.publish(RedisConfig.CHANNEL_NEWS, {
                "type":      "news_sentiment",
                "sentiment": cat.get("sentiment", "bullish"),
                "score":     cat.get("score", 0.9),
                "symbols":   cat.get("symbols", []),
                "catalyst":  cat.get("catalyst", ""),
                "title":     cat.get("title", ""),
                "priority":  "high",
                "source":    "morning_catalyst_scan",
                "timestamp": datetime.utcnow().isoformat(),
            })
            self.log(
                f"HIGH-priority catalyst: {cat.get('symbols')} | "
                f"{cat.get('catalyst')} | {cat.get('sentiment')}"
            )
        self.log(f"Catalyst scan complete | {len(catalysts)} events")

    async def _identify_catalysts(self, articles: list[dict]) -> list[dict]:
        items = [{"i": i, "t": a.get("title", "")[:90]}
                 for i, a in enumerate(articles[:30])]
        ck = LLMRouter.cache_key(items)
        prompt = (
            "Find catalysts: earnings_beat/miss, analyst_upgrade/downgrade, "
            "fda_approval/rejection, merger. Titles only:\n"
            f"{LLMRouter.j(items)}\n"
            "JSON:{\"catalysts\":[{\"symbols\":[],\"catalyst\":\"\","
            "\"sentiment\":\"bullish\",\"score\":0.9,\"title\":\"\"}]}"
        )
        raw = await self._router.call(
            [{"role": "user", "content": prompt}],
            prefer="haiku", max_tokens=800, cache_key=ck,
        )
        try:
            return json.loads(self._extract_json_block(raw)).get("catalysts", [])
        except Exception as e:
            self.log(f"Catalyst identification error: {e}", "warning")
            return []

    # ── Article publishing ─────────────────────────────────────────────────────

    async def _publish_scored(self, scored: list[dict], mode: str):
        bullish = bearish = 0
        for article in scored:
            s = article.get("sentiment", "neutral")
            if s == "bullish":
                bullish += 1
            elif s == "bearish":
                bearish += 1
            await self.publish(RedisConfig.CHANNEL_NEWS, {
                "type":   "news_sentiment",
                "mode":   mode,
                **article,
            })
        summary = {
            "type":           "news_summary",
            "articles_count": len(scored),
            "bullish":        bullish,
            "bearish":        bearish,
            "neutral":        len(scored) - bullish - bearish,
            "market_mood":    "bullish" if bullish > bearish
                               else "bearish" if bearish > bullish else "neutral",
            "mode":           mode,
            "timestamp":      datetime.utcnow().isoformat(),
        }
        await self.save_state("latest_summary", summary, ttl=300)
        self.log(
            f"[{mode}] published {len(scored)} | bull={bullish} bear={bearish}"
        )

    # ── Dedup helper ───────────────────────────────────────────────────────────

    def _dedup(self, articles: list[dict]) -> list[dict]:
        out = []
        for a in articles:
            uid = a.get("_id") or a.get("url", "")
            if uid and uid not in self._seen_ids:
                self._seen_ids.add(uid)
                out.append(a)
        if len(self._seen_ids) > 2000:
            self._seen_ids = set(list(self._seen_ids)[-1000:])
        return out

    # ── News fetchers (unchanged) ──────────────────────────────────────────────

    async def _fetch_alpaca_news(self, symbols: list[str], hours: int = 4) -> list[dict]:
        from config import AlpacaConfig
        url     = "https://data.alpaca.markets/v1beta1/news"
        params  = {"symbols": ",".join(symbols[:10]), "limit": 20, "sort": "desc"}
        headers = {
            "APCA-API-KEY-ID":     AlpacaConfig.API_KEY,
            "APCA-API-SECRET-KEY": AlpacaConfig.SECRET_KEY,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    return [
                        {
                            "title":       item.get("headline", ""),
                            "description": item.get("summary", ""),
                            "url":         item.get("url", ""),
                            "publishedAt": item.get("created_at", ""),
                            "_id":         str(item.get("id", "")),
                            "_symbols":    item.get("symbols", []),
                        }
                        for item in data.get("news", [])
                    ]
        except Exception as e:
            self.log(f"Alpaca news error: {e}", "warning")
            return []

    async def _fetch_newsapi(self, symbols: list[str], hours: int = 4) -> list[dict]:
        if not self.news_api:
            return []
        url    = "https://newsapi.org/v2/everything"
        params = {
            "q":        " OR ".join(symbols[:5]),
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 20,
            "from":     (datetime.now(timezone.utc) - timedelta(hours=hours))
                        .strftime("%Y-%m-%dT%H:%M:%S"),
            "apiKey":   self.news_api,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return []
                    return (await resp.json()).get("articles", [])
        except Exception as e:
            self.log(f"NewsAPI error: {e}", "warning")
            return []

    async def _fetch_yahoo_rss(self) -> list[dict]:
        import xml.etree.ElementTree as ET
        rss_urls = [
            "https://finance.yahoo.com/rss/topstories",
            "https://finance.yahoo.com/news/rssindex",
        ]
        articles: list[dict] = []
        seen: set[str]       = set()
        async with aiohttp.ClientSession() as session:
            for url in rss_urls:
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers={"User-Agent": "Mozilla/5.0 (compatible; tradingbot/1.0)"},
                    ) as resp:
                        if resp.status != 200:
                            continue
                        root = ET.fromstring(await resp.text())
                        for item in root.findall(".//item"):
                            title = (item.findtext("title") or "").strip()
                            link  = (item.findtext("link")  or "").strip()
                            pub   = (item.findtext("pubDate") or "").strip()
                            if not title or title in seen:
                                continue
                            seen.add(title)
                            articles.append({
                                "title":       title,
                                "url":         link,
                                "publishedAt": pub,
                                "_id":         f"yahoo_{abs(hash(title)) % 10 ** 9}",
                                "_symbols":    [],
                            })
                except Exception as e:
                    self.log(f"Yahoo RSS error ({url}): {e}", "warning")
        return articles

    # ── Keyword scorer (normal mode — no Claude) ───────────────────────────────

    _BULLISH_WORDS = {
        "surge", "rally", "soar", "jump", "beat", "upgrade", "buy",
        "record", "profit", "gain", "rise", "high", "boom", "strong",
        "growth", "positive", "outperform", "bullish", "optimistic",
    }
    _BEARISH_WORDS = {
        "drop", "fall", "crash", "miss", "downgrade", "sell", "loss",
        "decline", "plunge", "weak", "low", "risk", "concern", "bearish",
        "cut", "layoff", "bankrupt", "recession", "warning", "fears",
    }

    def _keyword_score_one(self, title: str) -> dict:
        words = set(re.findall(r"\b\w+\b", title.lower()))
        bull  = len(words & self._BULLISH_WORDS)
        bear  = len(words & self._BEARISH_WORDS)
        if bull > bear:
            return {"sentiment": "bullish", "score": 0.65}
        if bear > bull:
            return {"sentiment": "bearish", "score": 0.65}
        return {"sentiment": "neutral", "score": 0.5}

    def _keyword_score_articles(self, articles: list[dict]) -> list[dict]:
        watchlist_set = set(sym.upper() for sym in UniverseConfig.WATCHLIST)
        scored = []
        for article in articles:
            title = article.get("title", "")
            kw    = self._keyword_score_one(title)
            syms  = list(set(re.findall(r"\b[A-Z]{1,5}\b", title)) & watchlist_set)
            scored.append({
                "title":     title,
                "url":       article.get("url", ""),
                "published": article.get("publishedAt", ""),
                "sentiment": kw["sentiment"],
                "score":     kw["score"],
                "symbols":   syms or article.get("_symbols", []),
                "catalyst":  "",
            })
        non_neutral = sum(1 for s in scored if s["sentiment"] != "neutral")
        self.log(f"Keyword scored {len(scored)} | non-neutral={non_neutral}")
        return scored

    # ── JSON helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json_block(text: str) -> str:
        for start_ch, end_ch in [('{', '}'), ('[', ']')]:
            start = text.find(start_ch)
            if start == -1:
                continue
            depth, in_str, escape = 0, False, False
            for i, ch in enumerate(text[start:], start):
                if escape:
                    escape = False; continue
                if ch == '\\' and in_str:
                    escape = True; continue
                if ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == start_ch:
                        depth += 1
                    elif ch == end_ch:
                        depth -= 1
                        if depth == 0:
                            return text[start:i + 1]
        return text


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = NewsSentimentBot()
    asyncio.run(bot.start())
