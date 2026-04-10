"""
Bot 1 – News Sentiment Bot
Model  : Haiku 4.5  (fast, cheap, high-frequency)
Role   : Continuously fetches financial news headlines for watchlist symbols,
         scores sentiment with Haiku, and publishes scored events to the bus.
         Downstream bots (Strategy Agent, Master Commander) consume these signals.
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


POLL_INTERVAL = 1200  # seconds between news fetches (20 min)
MARKET_TZ     = pytz.timezone("America/New_York")


class NewsSentimentBot(BaseBot):
    """
    Bot 1 – News Sentiment Bot
    Fetches headlines via NewsAPI (or Alpaca news feed as fallback),
    scores them with Haiku, and publishes to CHANNEL_NEWS.
    """

    BOT_ID = 1
    NAME   = "News Sentiment Bot"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.HAIKU)
        self.client     = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self.news_api   = AlertConfig.NEWS_API_KEY
        self._seen_ids: set[str] = set()   # dedup headline IDs
        self._catalyst_scan_done: bool = False  # runs once at 9am per day

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        self.log(f"News Sentiment Bot starting | symbols={len(UniverseConfig.WATCHLIST)}")
        asyncio.create_task(self._morning_catalyst_task())

    async def run(self):
        while self.running:
            try:
                await self._news_cycle()
            except Exception as e:
                self.log(f"News cycle error: {e}", "error")
            await asyncio.sleep(POLL_INTERVAL)

    async def cleanup(self):
        self.log("News Sentiment Bot stopped")

    # ── 9am Catalyst scan (runs ONCE per day) ─────────────────────────────────

    async def _morning_catalyst_task(self):
        """At 9:00am EST, run a targeted catalyst scan and flag HIGH-priority trades."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now_et).total_seconds())
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            self._catalyst_scan_done = False
            await self._run_catalyst_scan()
            self._catalyst_scan_done = True

    async def _run_catalyst_scan(self):
        """
        Fetch pre-market news and flag stocks with HIGH-priority catalysts:
          - Earnings beat / miss
          - Analyst upgrade / downgrade
          - FDA approval / rejection
          - Merger / acquisition announcement
        Publishes each catalyst as a HIGH-priority news_sentiment event.
        """
        self.log("9am catalyst scan starting...")
        symbols = list(dict.fromkeys(
            UniverseConfig.WATCHLIST + getattr(UniverseConfig, "SCAN_UNIVERSE", [])
        ))

        # Fetch last 12 hours of news (captures overnight + pre-market)
        articles = await self._fetch_alpaca_news(symbols, hours=12)
        if not articles and self.news_api:
            articles = await self._fetch_newsapi(symbols, hours=12)

        if not articles:
            self.log("Catalyst scan: no articles found")
            return

        # Use Claude to identify high-priority catalysts
        catalysts = await self._identify_catalysts(articles)
        if not catalysts:
            self.log("Catalyst scan: no high-priority catalysts found")
            return

        # Publish each catalyst as HIGH priority
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

        self.log(f"Catalyst scan complete | {len(catalysts)} high-priority events")

    async def _identify_catalysts(self, articles: list[dict]) -> list[dict]:
        """Use Haiku to identify earnings/analyst/FDA/merger events from article list."""
        items = [
            {"title": a.get("title", ""), "summary": (a.get("description") or "")[:200]}
            for a in articles[:50]   # cap to 50 to keep prompt size reasonable
        ]
        prompt = (
            "You are a financial news analyst scanning pre-market news for high-priority catalysts.\n\n"
            "Identify articles about:\n"
            "  - Earnings beats or misses (EPS surprise)\n"
            "  - Analyst upgrades or downgrades (rating changes, price target changes)\n"
            "  - FDA approvals, rejections, or clinical trial results\n"
            "  - Merger, acquisition, or buyout announcements\n\n"
            f"Articles:\n{json.dumps(items)}\n\n"
            "For each high-priority event found, extract:\n"
            "  symbols (list of tickers affected), catalyst (type: earnings_beat/earnings_miss/"
            "analyst_upgrade/analyst_downgrade/fda_approval/fda_rejection/merger), "
            "sentiment (bullish/bearish), score (0.7-1.0), title (the headline)\n\n"
            "ONLY include events with clear catalyst types listed above. Ignore general news.\n"
            "Respond ONLY with JSON:\n"
            "{\"catalysts\": [{\"symbols\": [\"AAPL\"], \"catalyst\": \"earnings_beat\", "
            "\"sentiment\": \"bullish\", \"score\": 0.92, \"title\": \"...\"}]}"
        )
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw    = self._extract_json_block(response.content[0].text.strip())
            parsed = json.loads(raw)
            return parsed.get("catalysts", [])
        except Exception as e:
            self.log(f"Catalyst identification error: {e}", "warning")
            return []

    # ── News fetch ─────────────────────────────────────────────────────────────

    async def _fetch_newsapi(self, symbols: list[str], hours: int = 4) -> list[dict]:
        """Fetch headlines from NewsAPI.org."""
        if not self.news_api:
            return []

        query = " OR ".join(symbols[:5])   # API limit: keep query short
        url   = "https://newsapi.org/v2/everything"
        params = {
            "q":          query,
            "language":   "en",
            "sortBy":     "publishedAt",
            "pageSize":   20,
            "from":       (datetime.now(timezone.utc) - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%S"),
            "apiKey":     self.news_api,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    self.log(f"NewsAPI error {resp.status}", "warning")
                    return []
                data = await resp.json()
                return data.get("articles", [])

    async def _fetch_alpaca_news(self, symbols: list[str], hours: int = 4) -> list[dict]:
        """Fetch news from Alpaca's news endpoint (no API key beyond trading key)."""
        from config import AlpacaConfig
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": ",".join(symbols[:10]),
            "limit":   20,
            "sort":    "desc",
        }
        headers = {
            "APCA-API-KEY-ID":     AlpacaConfig.API_KEY,
            "APCA-API-SECRET-KEY": AlpacaConfig.SECRET_KEY,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    self.log(f"Alpaca news error {resp.status}", "warning")
                    return []
                data = await resp.json()
                # Normalise to same shape as NewsAPI
                articles = []
                for item in data.get("news", []):
                    articles.append({
                        "title":       item.get("headline", ""),
                        "description": item.get("summary", ""),
                        "url":         item.get("url", ""),
                        "publishedAt": item.get("created_at", ""),
                        "_id":         str(item.get("id", "")),
                        "_symbols":    item.get("symbols", []),
                    })
                return articles

    # ── Sentiment scoring ──────────────────────────────────────────────────────

    async def _score_articles(self, articles: list[dict]) -> list[dict]:
        """Score a batch of headlines with Haiku in a single API call."""
        if not articles:
            return []

        items = [
            {"id": i, "title": a.get("title", ""), "summary": a.get("description", "")[:300]}
            for i, a in enumerate(articles)
        ]

        prompt = (
            "You are a financial news sentiment analyser. "
            "For each article, output a sentiment score and the tickers most affected.\n\n"
            f"Articles: {json.dumps(items)}\n\n"
            "Respond ONLY with JSON:\n"
            "{\"results\": [{\"id\": 0, \"sentiment\": \"bullish|bearish|neutral\", "
            "\"score\": 0.85, \"symbols\": [\"AAPL\"], \"catalyst\": \"earnings beat\"}]}\n"
            "score is 0-1 confidence. catalyst is a short phrase (≤5 words)."
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=3000,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()

            # ── Robust JSON cleaning ────────────────────────────────────────
            # 1. Strip all markdown code fences
            raw = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
            # 2. Remove // line comments and /* block comments */
            raw = re.sub(r"//[^\n]*", "", raw)
            raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
            # 3. Fix Python/JS literals
            raw = raw.replace("None", "null").replace("True", "true").replace("False", "false")
            # 4. Remove trailing commas (two passes for nested structures)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            # 5. Bracket-tracking extraction of first complete {...} or [...]
            raw = self._extract_json_block(raw)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return []   # truncated or malformed — silent, caller gets []

            # Normalise: AI sometimes returns bare list instead of {"results":[...]}
            if isinstance(parsed, list):
                parsed = {"results": parsed}
            if not isinstance(parsed, dict):
                return []

            raw_results = parsed.get("results", [])
            if not isinstance(raw_results, list):
                return []

            # Build id→result map, guarding against non-dict items
            results = {
                r["id"]: r
                for r in raw_results
                if isinstance(r, dict) and "id" in r
            }

            scored = []
            for i, article in enumerate(articles):
                if not isinstance(article, dict):
                    continue
                r = results.get(i, {})
                if not isinstance(r, dict):
                    r = {}
                scored.append({
                    "title":     article.get("title", ""),
                    "url":       article.get("url", ""),
                    "published": article.get("publishedAt", ""),
                    "sentiment": r.get("sentiment", "neutral"),
                    "score":     r.get("score", 0.5),
                    "symbols":   r.get("symbols", article.get("_symbols", [])),
                    "catalyst":  r.get("catalyst", ""),
                })
            return scored

        except Exception as e:
            self.log(f"Sentiment scoring error: {e}", "warning")
            return []

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Bracket-tracking extractor: finds the first complete {...} or [...]
        block, correctly handles nested structures and quoted strings.
        Falls back to original text if no complete block found.
        """
        for start_ch, end_ch in [('{', '}'), ('[', ']')]:
            start = text.find(start_ch)
            if start == -1:
                continue
            depth = 0
            in_str = False
            escape = False
            for i, ch in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
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

    # ── Main cycle ─────────────────────────────────────────────────────────────

    async def _news_cycle(self):
        symbols = list(dict.fromkeys(UniverseConfig.WATCHLIST))

        # Try Alpaca news first (no extra key needed), fallback to NewsAPI
        articles = await self._fetch_alpaca_news(symbols, hours=4)
        if not articles and self.news_api:
            articles = await self._fetch_newsapi(symbols, hours=4)

        # Deduplicate
        new_articles = []
        for a in articles:
            uid = a.get("_id") or a.get("url", "")
            if uid and uid not in self._seen_ids:
                self._seen_ids.add(uid)
                new_articles.append(a)
        # Keep seen set bounded
        if len(self._seen_ids) > 2000:
            self._seen_ids = set(list(self._seen_ids)[-1000:])

        if not new_articles:
            self.log("No new articles")
            return

        scored = await self._score_articles(new_articles)

        # Publish each scored article
        bullish_count = 0
        bearish_count = 0
        for article in scored:
            sentiment = article.get("sentiment", "neutral")
            if sentiment == "bullish":
                bullish_count += 1
            elif sentiment == "bearish":
                bearish_count += 1

            await self.publish(RedisConfig.CHANNEL_NEWS, {
                "type":      "news_sentiment",
                **article,
            })

        # Summary state for Master Commander
        summary = {
            "type":          "news_summary",
            "articles_count": len(scored),
            "bullish":        bullish_count,
            "bearish":        bearish_count,
            "neutral":        len(scored) - bullish_count - bearish_count,
            "market_mood":   "bullish" if bullish_count > bearish_count
                              else "bearish" if bearish_count > bullish_count
                              else "neutral",
            "timestamp":     datetime.utcnow().isoformat(),
        }
        await self.save_state("latest_summary", summary, ttl=300)
        self.log(
            f"Scored {len(scored)} articles | "
            f"bull={bullish_count} bear={bearish_count}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = NewsSentimentBot()
    asyncio.run(bot.start())
