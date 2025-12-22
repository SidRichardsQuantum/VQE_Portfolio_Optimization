from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Iterable, Tuple, Optional, Literal

Freq = Literal["D","W","M"]

def _ensure_list(tickers: Iterable[str]) -> list[str]:
    return list(tickers)

def _to_adj_close(df: pd.DataFrame | pd.Series, tickers: list[str]) -> pd.DataFrame:
    # Handle 1 or many tickers, auto_adjust on/off
    if isinstance(df, pd.Series):
        # Single ticker; prefer Adj Close if present
        if df.name == "Adj Close":
            out = df.to_frame(name=tickers[0])
        elif df.name == "Close":
            out = df.to_frame(name=tickers[0])
        else:
            # yfinance single-ticker returns a column per field
            if "Adj Close" in df.columns:
                out = df["Adj Close"].to_frame(name=tickers[0])
            else:
                out = df["Close"].to_frame(name=tickers[0])
        return out

    # MultiIndex case
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(-1)):
            out = df.xs("Adj Close", axis=1, level=-1)
        else:
            out = df.xs("Close", axis=1, level=-1)
        # keep only requested tickers and in the same order
        out = out.reindex(columns=tickers)
    else:
        # Already a wide DataFrame of prices
        out = df.copy()
        out = out.reindex(columns=tickers, fill_value=np.nan)
    return out.astype("float64")

def _infer_annualization(index: pd.DatetimeIndex) -> Tuple[int, Freq]:
    # Try frequency; fall back to spacing
    freq = pd.infer_freq(index)
    if freq and freq.startswith("B") or freq == "D":
        return 252, "D"
    if freq and freq.startswith("W"):
        return 52, "W"
    if freq and freq.startswith("M"):
        return 12, "M"
    # Fallback based on median spacing
    deltas = np.median(np.diff(index.values).astype("timedelta64[D]").astype(int))
    if deltas <= 2:
        return 252, "D"
    if deltas <= 8:
        return 52, "W"
    return 12, "M"

def fetch_prices(
    tickers: Iterable[str],
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """Download adjusted prices for tickers on a business-day index."""
    t = _ensure_list(tickers)
    raw = yf.download(t, start=start, end=end, auto_adjust=auto_adjust, group_by="ticker", progress=progress, threads=True)
    prices = _to_adj_close(raw, t)

    # Standardize to business days and forward-fill short gaps
    bidx = pd.bdate_range(prices.index.min(), prices.index.max())
    prices = prices.reindex(bidx).ffill(limit=5)  # limit prevents long stale fills
    return prices

def compute_mu_sigma(
    prices: pd.DataFrame,
    use_log: bool = True,
    shrink: Optional[Literal["lw"]] = None,
    scale: Optional[Literal["none","trace","max"]] = "none",
) -> tuple[pd.Series, pd.DataFrame, int]:
    """Annualized mean vector and covariance matrix with options."""
    ret = np.log(prices).diff().dropna() if use_log else prices.pct_change().dropna()

    af, _ = _infer_annualization(prices.index)
    mu = (ret.mean() * af).astype("float64")

    if shrink == "lw":
        try:
            from sklearn.covariance import LedoitWolf
            Sigma = pd.DataFrame(LedoitWolf().fit(ret.values).covariance_, index=ret.columns, columns=ret.columns) * af
        except Exception:
            Sigma = (ret.cov() * af).astype("float64")
    else:
        Sigma = (ret.cov() * af).astype("float64")

    if scale == "trace":
        tr = float(np.trace(Sigma.values))
        if tr > 0:
            Sigma = Sigma / tr
    elif scale == "max":
        m = float(np.max(np.abs(Sigma.values)))
        if m > 0:
            Sigma = Sigma / m
    # mu left unscaled by default to preserve risk/return ratio

    return mu, Sigma, af

def get_stock_data(
    tickers: Iterable[str],
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    auto_adjust: bool = True,
    use_log: bool = True,
    shrink: Optional[Literal["lw"]] = None,
    scale: Optional[Literal["none","trace","max"]] = "none",
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper. Returns (mu, Sigma, prices).
    mu: pd.Series indexed by ticker
    Sigma: pd.DataFrame indexed by ticker x ticker
    prices: pd.DataFrame of adjusted prices
    """
    prices = fetch_prices(tickers, start, end, auto_adjust=auto_adjust)
    mu, Sigma, _ = compute_mu_sigma(prices, use_log=use_log, shrink=shrink, scale=scale)
    return mu, Sigma, prices

if __name__ == "__main__":
    tickers = ["AAPL","MSFT","GOOGL","AMZN"]
    mu, Sigma, prices = get_stock_data(tickers)
    print(mu.round(4))
    print(Sigma.round(6))
    print(prices.tail())
