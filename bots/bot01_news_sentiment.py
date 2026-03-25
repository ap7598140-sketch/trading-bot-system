"""
Bot 1 – News Sentiment Bot
Model  : Haiku 4.5  (fast, cheap, high-frequency)
Role   : Continuously fetches financial news headlines for watchlist symbols,
         scores sentiment with Haiku, and publishes scored events to the bus.
         Downstream bots (Strategy Agent, Master Commander) consume these signals.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional
import aiohttp

import anthropic

from config import Models, RedisConfig, UniverseConfig, AnthropicConfig, AlertConfig
from shared.base_bot import BaseBot


POLL_INTERVAL = 120   # seconds between news fetches


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

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        self.log(f"News Sentiment Bot starting | symbols={len(UniverseConfig.WATCHLIST)}")

    async def run(self):
        while self.running:
            try:
                await self._news_cycle()
            except Exception as e:
                self.log(f"News cycle error: {e}", "error")
            await asyncio.sleep(POLL_INTERVAL)

    async def cleanup(self):
        self.log("News Sentiment Bot stopped")

    # ── News fetch ─────────────────────────────────────────────────────────────

    async def _fetch_newsapi(self, symbols: list[str]) -> list[dict]:
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

    async def _fetch_alpaca_news(self, symbols: list[str]) -> list[dict]:
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
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed  = json.loads(raw)
            results = {r["id"]: r for r in parsed.get("results", [])}

            scored = []
            for i, article in enumerate(articles):
                r = results.get(i, {})
                scored.append({
                    "title":      article.get("title", ""),
                    "url":        article.get("url", ""),
                    "published":  article.get("publishedAt", ""),
                    "sentiment":  r.get("sentiment", "neutral"),
                    "score":      r.get("score", 0.5),
                    "symbols":    r.get("symbols", article.get("_symbols", [])),
                    "catalyst":   r.get("catalyst", ""),
                })
            return scored

        except Exception as e:
            self.log(f"Sentiment scoring error: {e}", "warning")
            return []

    # ── Main cycle ─────────────────────────────────────────────────────────────

    async def _news_cycle(self):
        symbols = list(dict.fromkeys(UniverseConfig.WATCHLIST))

        # Try Alpaca news first (no extra key needed), fallback to NewsAPI
        articles = await self._fetch_alpaca_news(symbols)
        if not articles and self.news_api:
            articles = await self._fetch_newsapi(symbols)

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
