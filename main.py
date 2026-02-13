import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# =========================
# KEYS (ENV)
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("futures-analytics-bot")

MAIN_MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📊 Зроби прогноз")],
        [KeyboardButton(text="❌ Скасувати")],
    ],
    resize_keyboard=True,
)


@dataclass
class TradeSignal:
    symbol: str
    probability: int
    entry: float
    take_profit_1: float
    take_profit_2: float
    stop_loss: float
    timeframe: str
    trend: str


class FuturesAnalyzer:
    def __init__(self) -> None:
        self.client: AsyncClient | None = None

    async def connect(self) -> None:
        self.client = await AsyncClient.create(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_API_SECRET,
        )
        logger.info("Connected to Binance Futures API")

    async def close(self) -> None:
        if self.client:
            await self.client.close_connection()
            logger.info("Closed Binance API connection")

    async def _get_usdt_perpetual_symbols(self) -> list[str]:
        if not self.client:
            return []
        exchange_info = await self.client.futures_exchange_info()
        symbols = [
            s["symbol"]
            for s in exchange_info.get("symbols", [])
            if s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
        ]
        tickers = await self.client.futures_ticker()
        volume_map = {
            t["symbol"]: float(t.get("quoteVolume", 0.0))
            for t in tickers
            if t.get("symbol") in symbols
        }
        symbols.sort(key=lambda s: volume_map.get(s, 0.0), reverse=True)
        return symbols[:80]

    async def _klines_df(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        if not self.client:
            raise RuntimeError("Binance client is not connected")
        klines = await self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            raise ValueError(f"No kline data for {symbol} {interval}")

        frame = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = frame[col].astype(float)
        return frame

    async def _get_funding_rate(self, symbol: str) -> float:
        if not self.client:
            return 0.0
        rates = await self.client.futures_funding_rate(symbol=symbol, limit=1)
        if not rates:
            return 0.0
        return float(rates[-1].get("fundingRate", 0.0))

    async def _get_open_interest(self, symbol: str) -> float:
        if not self.client:
            return 0.0
        oi = await self.client.futures_open_interest(symbol=symbol)
        return float(oi.get("openInterest", 0.0))

    def _score_signal(
        self,
        df15: pd.DataFrame,
        df1h: pd.DataFrame,
        funding_rate: float,
        open_interest: float,
    ) -> tuple[int, str]:
        close_15 = df15["close"]
        close_1h = df1h["close"]
        vol_15 = df15["volume"]

        ema50_15 = EMAIndicator(close_15, window=50).ema_indicator().iloc[-1]
        ema200_15 = EMAIndicator(close_15, window=200).ema_indicator().iloc[-1]
        ema50_1h = EMAIndicator(close_1h, window=50).ema_indicator().iloc[-1]
        ema200_1h = EMAIndicator(close_1h, window=200).ema_indicator().iloc[-1]
        rsi_15 = RSIIndicator(close_15, window=14).rsi().iloc[-1]
        rsi_1h = RSIIndicator(close_1h, window=14).rsi().iloc[-1]

        current_price = close_15.iloc[-1]
        avg_volume = vol_15.iloc[-40:-1].mean()
        current_volume = vol_15.iloc[-1]

        ema_bias = 1 if (ema50_15 > ema200_15 and ema50_1h > ema200_1h) else -1
        rsi_bias = 1 if (53 <= rsi_15 <= 72 and 50 <= rsi_1h <= 70) else -1
        volume_bias = 1 if current_volume > avg_volume * 1.2 else -1
        funding_bias = 1 if -0.0005 <= funding_rate <= 0.001 else -1
        oi_bias = 1 if open_interest > 0 else -1
        momentum_bias = 1 if current_price > close_15.iloc[-4] else -1

        weights = {
            "ema": 0.30,
            "rsi": 0.20,
            "volume": 0.15,
            "funding": 0.10,
            "open_interest": 0.10,
            "momentum": 0.15,
        }

        weighted = (
            (1 if ema_bias > 0 else 0) * weights["ema"]
            + (1 if rsi_bias > 0 else 0) * weights["rsi"]
            + (1 if volume_bias > 0 else 0) * weights["volume"]
            + (1 if funding_bias > 0 else 0) * weights["funding"]
            + (1 if oi_bias > 0 else 0) * weights["open_interest"]
            + (1 if momentum_bias > 0 else 0) * weights["momentum"]
        )

        probability = int(round(weighted * 100))
        trend = "bullish" if ema_bias > 0 and current_price > ema50_15 else "bearish"
        return probability, trend

    def _build_trade_levels(self, close_price: float, trend: str, df15: pd.DataFrame) -> tuple[float, float, float, float]:
        atr_proxy = float((df15["high"] - df15["low"]).tail(14).mean())
        atr_proxy = max(atr_proxy, close_price * 0.003)

        if trend == "bullish":
            entry = close_price
            tp1 = close_price + atr_proxy * 1.1
            tp2 = close_price + atr_proxy * 2.4
            sl = close_price - atr_proxy * 0.9
        else:
            entry = close_price
            tp1 = close_price - atr_proxy * 1.1
            tp2 = close_price - atr_proxy * 2.4
            sl = close_price + atr_proxy * 0.9

        digits = 4 if close_price < 1 else 2
        return (
            round(entry, digits),
            round(tp1, digits),
            round(tp2, digits),
            round(sl, digits),
        )

    async def find_signal(self) -> TradeSignal | None:
        if not self.client:
            raise RuntimeError("Analyzer is not initialized")

        symbols = await self._get_usdt_perpetual_symbols()
        logger.info("Scanning %s symbols", len(symbols))

        for symbol in symbols:
            try:
                df15, df1h = await asyncio.gather(
                    self._klines_df(symbol, AsyncClient.KLINE_INTERVAL_15MINUTE),
                    self._klines_df(symbol, AsyncClient.KLINE_INTERVAL_1HOUR),
                )
                funding_rate, open_interest = await asyncio.gather(
                    self._get_funding_rate(symbol),
                    self._get_open_interest(symbol),
                )

                probability, trend = self._score_signal(df15, df1h, funding_rate, open_interest)
                logger.info(
                    "Analyzed %s | probability=%s%% trend=%s funding=%s oi=%s",
                    symbol,
                    probability,
                    trend,
                    funding_rate,
                    open_interest,
                )

                if probability < 85:
                    continue

                close_price = float(df15["close"].iloc[-1])
                entry, tp1, tp2, sl = self._build_trade_levels(close_price, trend, df15)

                return TradeSignal(
                    symbol=symbol,
                    probability=probability,
                    entry=entry,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    stop_loss=sl,
                    timeframe="15m",
                    trend=trend,
                )
            except BinanceAPIException as api_err:
                logger.warning("Binance API error for %s: %s", symbol, api_err)
            except Exception as err:
                logger.exception("Unexpected analysis error for %s: %s", symbol, err)

        return None


def format_signal(signal: TradeSignal) -> str:
    return (
        "📈 ПРОГНОЗ ФʼЮЧЕРСІВ\n\n"
        f"Пара: {signal.symbol}\n"
        f"Ймовірність: {signal.probability}%\n\n"
        f"Вхід: {signal.entry}\n"
        f"Take Profit 1: {signal.take_profit_1}\n"
        f"Take Profit 2: {signal.take_profit_2}\n"
        f"Stop Loss: {signal.stop_loss}\n\n"
        f"Таймфрейм: {signal.timeframe}\n"
        f"Тренд: {signal.trend}"
    )


async def main() -> None:
    analyzer = FuturesAnalyzer()
    await analyzer.connect()

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        await message.answer("Обери дію з меню:", reply_markup=MAIN_MENU)

    @dp.message(F.text == "❌ Скасувати")
    async def cancel_action(message: Message) -> None:
        await message.answer("Поточну дію скасовано.", reply_markup=MAIN_MENU)

    @dp.message(F.text == "📊 Зроби прогноз")
    async def make_forecast(message: Message) -> None:
        await message.answer("Аналізую ринок Binance Futures, зачекай...")
        try:
            signal = await analyzer.find_signal()
            if signal is None:
                await message.answer(
                    "Наразі немає сигналів з ймовірністю 85%+ . Спробуй пізніше.",
                    reply_markup=MAIN_MENU,
                )
                return
            await message.answer(format_signal(signal), reply_markup=MAIN_MENU)
        except Exception as err:
            logger.exception("Forecast processing failed: %s", err)
            await message.answer(
                "Сталася помилка під час аналізу. Спробуй ще раз пізніше.",
                reply_markup=MAIN_MENU,
            )

    try:
        await dp.start_polling(bot)
    finally:
        await analyzer.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
