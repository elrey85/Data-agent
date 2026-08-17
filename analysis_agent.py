import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnalysisAgent")


class AnalysisAgent:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder

    def load_data(self, symbol: str) -> pd.DataFrame:
        """Liest die CSV-Datei des Daten-Agenten ein."""
        path = os.path.join(self.data_folder, f"{symbol}_latest.csv")
        if not os.path.exists(path):
            logger.warning(f"Keine Datei gefunden: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    def calculate_sma(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        return df["close"].rolling(window=window).mean()

    def calculate_ema(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        return df["close"].ewm(span=window, adjust=False).mean()

    def calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: int = 2):
        sma = self.calculate_sma(df, window)
        std = df["close"].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def generate_signal(self, df: pd.DataFrame) -> str:
        """Sehr einfache Signal-Logik basierend auf RSI und SMA-Kreuzung."""
        if len(df) < 20:
            return "Nicht genug Daten"

        rsi = self.calculate_rsi(df).iloc[-1]
        sma_short = self.calculate_sma(df, window=10).iloc[-1]
        sma_long = self.calculate_sma(df, window=20).iloc[-1]

        signal = "Neutral"

        if rsi < 30 and sma_short > sma_long:
            signal = "Kaufsignal (überverkauft + Aufwärtstrend)"
        elif rsi > 70 and sma_short < sma_long:
            signal = "Verkaufssignal (überkauft + Abwärtstrend)"
        elif sma_short > sma_long:
            signal = "Leichter Aufwärtstrend"
        elif sma_short < sma_long:
            signal = "Leichter Abwärtstrend"

        return signal

    def analyze(self, symbol: str) -> dict:
        """Führt die komplette Analyse für ein Symbol durch."""
        df = self.load_data(symbol)
        if df.empty:
            return {"symbol": symbol, "error": "Keine Daten verfügbar"}

        df["sma_10"] = self.calculate_sma(df, 10)
        df["sma_20"] = self.calculate_sma(df, 20)
        df["ema_20"] = self.calculate_ema(df, 20)
        df["rsi_14"] = self.calculate_rsi(df, 14)

        macd_line, signal_line, histogram = self.calculate_macd(df)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram

        upper, lower = self.calculate_bollinger_bands(df)
        df["bb_upper"] = upper
        df["bb_lower"] = lower

        signal = self.generate_signal(df)

        result = {
            "symbol": symbol,
            "last_close": round(df["close"].iloc[-1], 2),
            "last_rsi": round(df["rsi_14"].iloc[-1], 2) if not pd.isna(df["rsi_14"].iloc[-1]) else None,
            "sma_10": round(df["sma_10"].iloc[-1], 2) if not pd.isna(df["sma_10"].iloc[-1]) else None,
            "sma_20": round(df["sma_20"].iloc[-1], 2) if not pd.isna(df["sma_20"].iloc[-1]) else None,
            "signal": signal
        }

        # Vollständige Analyse-Tabelle speichern
        os.makedirs("analysis", exist_ok=True)
        df.to_csv(f"analysis/{symbol}_analysis.csv")

        return result


if __name__ == "__main__":
    agent = AnalysisAgent()
    symbols = ["AAPL", "MSFT", "TSLA"]

    results = []
    for symbol in symbols:
        result = agent.analyze(symbol)
        results.append(result)
        print(f"\n--- {symbol} ---")
        for key, value in result.items():
            print(f"{key}: {value}")

    # Zusammenfassung speichern
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("analysis/summary.csv", index=False)
    print("\nZusammenfassung gespeichert: analysis/summary.csv")
