import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_sharpe_ratio(tickers, weights, risk_free_rate):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"Hamtar data for: {', '.join(tickers)}")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
    
    if isinstance(data, pd.Series):
        data = data.to_frame()
        data.columns = [tickers[0]]
    
    # Remove tickers that failed to download
    available_tickers = [t for t in tickers if t in data.columns]
    if len(available_tickers) < len(tickers):
        missing = set(tickers) - set(available_tickers)
        print(f"WARNING: Could not fetch data for: {missing}")
        print(f"Continuing with {len(available_tickers)} available tickers")
    
    if len(available_tickers) == 0:
        raise ValueError("No valid tickers found")
    
    # Adjust weights for available tickers
    ticker_indices = [i for i, t in enumerate(tickers) if t in available_tickers]
    adjusted_weights = np.array([weights[i] for i in ticker_indices])
    adjusted_weights = adjusted_weights / adjusted_weights.sum()  # Renormalize
    
    returns_list = []
    
    # Currency mapping (Yahoo Finance format)
    currency_pairs = {
        'SEK': 'SEKUSD=X',
        'EUR': 'EURUSD=X',
        'GBP': 'GBPUSD=X',
        'DKK': 'DKKUSD=X',
        'NOK': 'NOKUSD=X',
        'CHF': 'CHFUSD=X',
        'JPY': 'JPYUSD=X'
    }
    
    for ticker in available_tickers:
        print(f"Bearbetar {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            currency = stock_info.get('currency', 'USD')
        except:
            currency = 'USD'
            print(f"  Could not get currency info for {ticker}, assuming USD")
        
        if currency != 'USD' and currency in currency_pairs:
            print(f"  {ticker} ar i {currency}, konverterar till USD...")
            fx_ticker = currency_pairs[currency]
            try:
                fx_data = yf.download(fx_ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
                
                aligned_data = pd.DataFrame({
                    'asset': data[ticker],
                    'fx': fx_data
                }).ffill().dropna()
                
                asset_usd = aligned_data['asset'] * aligned_data['fx']
                asset_returns = asset_usd.pct_change(fill_method=None).dropna()
                
            except Exception as e:
                print(f"  Varning: Kunde inte hamta FX-data for {currency}. Antar USD.")
                asset_returns = data[ticker].pct_change(fill_method=None).dropna()
        else:
            print(f"  {ticker} ar redan i USD")
            asset_returns = data[ticker].pct_change(fill_method=None).dropna()
        
        returns_list.append(asset_returns)
    
    returns_df = pd.concat(returns_list, axis=1, join='inner')
    returns_df.columns = available_tickers
    
    if returns_df.empty:
        raise ValueError("No overlapping dates found for assets")
    
    portfolio_returns = (returns_df * adjusted_weights).sum(axis=1)
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    print(f"\nUsed {len(available_tickers)} out of {len(tickers)} tickers")
    print(f"Adjusted weights: {adjusted_weights}")
    
    return {
        'sharpe_ratio': float(sharpe_ratio),
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_volatility),
        'portfolio_returns': portfolio_returns.tolist()
    }