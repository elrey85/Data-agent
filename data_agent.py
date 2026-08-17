import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataAgent")


class DataAgent:
    def _init_(self):
        self.cache = {}

    def get_historical_data(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"Keine Daten für {symbol}")
                return pd.DataFrame()
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume"
            })
            df["symbol"] = symbol
            return df[["open", "high", "low", "close", "volume", "symbol"]]
        except Exception as e:
            logger.error(f"Fehler bei {symbol}: {e}")
            return pd.DataFrame()


if _name_ == "_main_":
    agent = DataAgent()
    symbols = ["AAPL", "MSFT", "TSLA"]

    for symbol in symbols:
        df = agent.get_historical_data(symbol, period="1mo")
        print(f"\n--- {symbol} ---")
        print(df.tail(5))

        # Ergebnis als CSV speichern
        df.to_csv(f"data/{symbol}_latest.csv")
        print(f"Gespeichert: data/{symbol}_latest.csv")
