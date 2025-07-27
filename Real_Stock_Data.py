import yfinance as yf
import pandas as pd
import numpy as np


def fetch_data(tickers, start="2023-01-01", end="2024-01-01"):
    """Downloads adjusted closing prices for given tickers and dates."""
    data = yf.download(tickers, start=start, end=end, group_by="ticker", progress=False)

    try:
        adj_close = pd.concat([data[ticker]['Adj Close'] for ticker in tickers], axis=1)
        adj_close.columns = tickers
    except Exception as e:
        print("Warning: Could not access 'Adj Close' as expected:", e)
        adj_close = data

    return adj_close


def compute_mu_and_sigma(prices, normalize=True):
    """Computes annualized mean returns and covariances from price data."""
    returns = prices.pct_change().dropna()  # Use adjusted close prices ONLY

    mu = returns.mean().values * 252  # Annualized mean return
    Sigma = returns.cov().values * 252  # Annualized covariance matrix

    if normalize:
        mu = mu / np.max(np.abs(mu))
        Sigma = Sigma / np.max(np.abs(Sigma))

    return mu, Sigma


def get_stock_data(tickers, start="2023-01-01", end="2024-01-01", normalize=True):
    """Convenience wrapper to get mu and Sigma."""
    prices = fetch_data(tickers, start, end)
    mu, Sigma = compute_mu_and_sigma(prices, normalize=normalize)
    return mu, Sigma, prices


# Optional: test call
# if __name__ == "__main__":
#     tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
#     mu, Sigma, adj_close = get_stock_data(tickers)
#     print("Prices:\n", adj_close.tail())
#     print("mu:\n", mu)
#     print("Sigma:\n", Sigma)
