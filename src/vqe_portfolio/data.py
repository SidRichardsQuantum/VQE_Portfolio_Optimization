# src/vqe_portfolio/data.py
from __future__ import annotations

from typing import Iterable, Optional, Literal, Tuple

import numpy as np

from .utils import ensure_list

Freq = Literal["D", "W", "M"]


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "vqe_portfolio.data requires optional dependencies. "
            "Install with: pip install 'vqe-portfolio[data]'"
        ) from e
    return pd


def _require_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "vqe_portfolio.data requires optional dependencies. "
            "Install with: pip install 'vqe-portfolio[data]'"
        ) from e
    return yf


def _to_adj_close(df, tickers: list[str]):
    """
    Extract an adjusted-close-like DataFrame from yfinance output.

    Notes:
    - Handles single-ticker Series return shapes and multi-ticker MultiIndex columns.
    - Prefers "Adj Close" when present, otherwise falls back to "Close".
    """
    pd = _require_pandas()

    # Handle 1 ticker Series
    if isinstance(df, pd.Series):
        if df.name == "Adj Close":
            return df.to_frame(name=tickers[0]).astype("float64")
        if df.name == "Close":
            return df.to_frame(name=tickers[0]).astype("float64")

        # Defensive fallback: some yfinance variants can return a Series-like
        # but with fields embedded; if not possible, just treat as close.
        return df.to_frame(name=tickers[0]).astype("float64")

    # MultiIndex columns (common for multiple tickers with group_by="ticker")
    if isinstance(getattr(df, "columns", None), pd.MultiIndex):
        lvl = df.columns.get_level_values(-1)
        field = "Adj Close" if "Adj Close" in lvl else "Close"
        out = df.xs(field, axis=1, level=-1)
        out = out.reindex(columns=tickers)
        return out.astype("float64")

    # Flat columns: attempt to reindex to tickers
    out = df.copy()
    out = out.reindex(columns=tickers, fill_value=np.nan)
    return out.astype("float64")


def _infer_annualization(index) -> Tuple[int, Freq]:
    pd = _require_pandas()

    freq = pd.infer_freq(index)
    if freq and (freq.startswith("B") or freq == "D"):
        return 252, "D"
    if freq and freq.startswith("W"):
        return 52, "W"
    if freq and freq.startswith("M"):
        return 12, "M"

    # Fallback: median spacing in days
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
):
    """
    Download adjusted prices for tickers on a business-day index.

    Requires: vqe-portfolio[data]
    """
    pd = _require_pandas()
    yf = _require_yfinance()

    t = ensure_list(tickers)
    raw = yf.download(
        t,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=progress,
        threads=True,
    )
    prices = _to_adj_close(raw, t)

    if prices.empty:
        raise ValueError("No price data returned. Check tickers and date range.")

    # Ensure a business-day index and forward fill short gaps
    bidx = pd.bdate_range(prices.index.min(), prices.index.max())
    prices = prices.reindex(bidx).ffill(limit=5)

    # Drop rows where all tickers are missing
    prices = prices.dropna(how="all")

    return prices


def compute_mu_sigma(
    prices,
    use_log: bool = True,
    shrink: Optional[Literal["lw"]] = None,
    scale: Optional[Literal["none", "trace", "max"]] = "none",
):
    """
    Annualized mean vector and covariance matrix with options.

    Parameters
    ----------
    prices:
        pandas.DataFrame of prices indexed by datetime.
    use_log:
        If True, use log returns; otherwise simple returns.
    shrink:
        If "lw", attempt Ledoit-Wolf shrinkage (requires scikit-learn),
        otherwise fall back to sample covariance.
    scale:
        "none" (natural units), "trace" (divide by trace), or "max" (divide by max abs entry).

    Returns
    -------
    (mu, Sigma, annualization_factor)
    """
    pd = _require_pandas()

    if prices is None:
        raise ValueError("prices must be a pandas DataFrame, got None")

    # Returns
    if use_log:
        # Guard against non-positive prices (log undefined)
        if (prices <= 0).any().any():
            raise ValueError("Non-positive price encountered; cannot compute log returns.")
        ret = np.log(prices).diff().dropna()
    else:
        ret = prices.pct_change().dropna()

    af, _ = _infer_annualization(prices.index)
    mu = (ret.mean() * af).astype("float64")

    # Covariance with optional shrinkage
    if shrink == "lw":
        try:
            from sklearn.covariance import LedoitWolf  # type: ignore

            Sigma = (
                pd.DataFrame(
                    LedoitWolf().fit(ret.values).covariance_,
                    index=ret.columns,
                    columns=ret.columns,
                )
                * af
            ).astype("float64")
        except Exception:
            Sigma = (ret.cov() * af).astype("float64")
    else:
        Sigma = (ret.cov() * af).astype("float64")

    # Optional scaling
    if scale == "trace":
        tr = float(np.trace(Sigma.values))
        if tr > 0:
            Sigma = Sigma / tr
    elif scale == "max":
        m = float(np.max(np.abs(Sigma.values)))
        if m > 0:
            Sigma = Sigma / m
    elif scale != "none":
        raise ValueError(f"scale must be one of 'none', 'trace', 'max'; got {scale}")

    return mu, Sigma, af


def get_stock_data(
    tickers: Iterable[str],
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    auto_adjust: bool = True,
    use_log: bool = True,
    shrink: Optional[Literal["lw"]] = None,
    scale: Optional[Literal["none", "trace", "max"]] = "none",
):
    """
    Convenience wrapper. Returns (mu, Sigma, prices).

    Requires: vqe-portfolio[data]
    """
    prices = fetch_prices(tickers, start, end, auto_adjust=auto_adjust)
    mu, Sigma, _ = compute_mu_sigma(prices, use_log=use_log, shrink=shrink, scale=scale)
    return mu, Sigma, prices
