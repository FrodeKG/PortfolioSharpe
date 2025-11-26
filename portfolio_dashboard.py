import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sharpe_calculator import calculate_sharpe_ratio

# Page config
st.set_page_config(page_title="Portfolio Analyzer", page_icon="📊", layout="wide")

st.title("📊 Portfolio Sharpe Ratio Analyzer")
st.markdown("---")

# Portfolio settings
tickers = ['FABG.ST','CAST.ST','BETS-B.ST','FFARM.AS','IVSO.ST','SGHC','BAHN-B.ST','MMM','FTS','SHB-B.ST','BNP.PA','DANSKE.CO','0P0001N5ZV.ST','0P00000LEI.ST','0P000151JW.F','0P0000YVZ3.ST','0P0001ECQR.ST','0P0001Q6FC.ST','0P00011IEP.ST']
weights = [0.1296, 0.1775, 0.0352, 0.0167, 0.0303, 0.0093, 0.0058, 0.0609, 0.0371, 0.0487, 0.0286, 0.0243, 0.0723, 0.067, 0.0933, 0.0866, 0.0451, 0.0256, 0.0061]
risk_free_rate = 0.041

# Fetch current prices
st.subheader("📈 Current Holdings")

with st.spinner("Fetching latest prices..."):
    holdings_data = []
    
    for ticker, weight in zip(tickers, weights):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                
                # Calculate change
                if len(hist) > 1:
                    prev_close = hist['Close'].iloc[-2]
                    change = ((current_price - prev_close) / prev_close * 100)
                else:
                    change = 0
                
                # Get name
                name = info.get('longName', info.get('shortName', ticker))
                
                # Get dividend yield if available
                div_yield = info.get('dividendYield', 0)
                if div_yield:
                    div_yield = div_yield * 100  # Convert to percentage
                
                holdings_data.append({
                    'Ticker': ticker,
                    'Name': name[:30] + '...' if len(name) > 30 else name,
                    'Weight (%)': round(weight * 100, 2),
                    'Current Price': round(current_price, 2),
                    'Currency': info.get('currency', 'N/A'),
                    'Change (%)': round(change, 2),
                    'Div Yield (%)': round(div_yield, 2) if div_yield else 0,
                    'Type': 'Fund' if ticker.startswith('0P') else 'Stock'
                })
            else:
                holdings_data.append({
                    'Ticker': ticker,
                    'Name': 'N/A',
                    'Weight (%)': round(weight * 100, 2),
                    'Current Price': 'N/A',
                    'Currency': 'N/A',
                    'Change (%)': 0,
                    'Div Yield (%)': 0,
                    'Type': 'Fund' if ticker.startswith('0P') else 'Unknown'
                })
        except Exception as e:
            holdings_data.append({
                'Ticker': ticker,
                'Name': f'Error: {str(e)[:20]}',
                'Weight (%)': round(weight * 100, 2),
                'Current Price': 'N/A',
                'Currency': 'N/A',
                'Change (%)': 0,
                'Div Yield (%)': 0,
                'Type': 'Fund' if ticker.startswith('0P') else 'Unknown'
            })

holdings_df = pd.DataFrame(holdings_data)

# Color code the Change column
def color_change(val):
    try:
        color = 'green' if float(val) > 0 else 'red' if float(val) < 0 else 'gray'
        return f'color: {color}'
    except:
        return ''

# Display with filters
col1, col2 = st.columns([1, 4])
with col1:
    filter_type = st.selectbox("Filter by type:", ["All", "Stock", "Fund"])

if filter_type != "All":
    filtered_df = holdings_df[holdings_df['Type'] == filter_type]
else:
    filtered_df = holdings_df

st.dataframe(
    filtered_df.style.applymap(color_change, subset=['Change (%)']),
    use_container_width=True,
    height=400
)

st.caption(f"Showing {len(filtered_df)} of {len(holdings_df)} holdings")

# 1-Month Performance Tracker
st.markdown("---")
st.subheader("📅 Sharp Portfolio - 1 Month Performance (Including Dividends)")

with st.spinner("Calculating 1-month performance with dividends..."):
    try:
        # Fetch 1 month of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=35)
        
        month_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if 'Adj Close' in month_data.columns.levels[0] if isinstance(month_data.columns, pd.MultiIndex) else True:
            prices = month_data['Adj Close'] if isinstance(month_data.columns, pd.MultiIndex) else month_data
        else:
            prices = month_data['Close'] if isinstance(month_data.columns, pd.MultiIndex) else month_data
        
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
            prices.columns = [tickers[0]]
        
        # Calculate daily portfolio value
        initial_investment = 10000
        
        # Get available tickers
        available_tickers = [t for t in tickers if t in prices.columns]
        ticker_indices = [i for i, t in enumerate(tickers) if t in available_tickers]
        adjusted_weights = np.array([weights[i] for i in ticker_indices])
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # Calculate total return including dividends
        total_returns_list = []
        dividends_paid = {}
        
        for ticker in available_tickers:
            try:
                stock = yf.Ticker(ticker)
                # Get dividends in the period
                divs = stock.dividends
                divs_in_period = divs[(divs.index >= start_date) & (divs.index <= end_date)]
                
                # Calculate price returns
                price_returns = prices[ticker].pct_change().fillna(0)
                
                # Add dividend returns on dividend dates
                total_returns = price_returns.copy()
                for div_date, div_amount in divs_in_period.items():
                    # Find closest trading day
                    closest_date = prices.index[prices.index.get_indexer([div_date], method='nearest')[0]]
                    if closest_date in total_returns.index:
                        price_on_div = prices[ticker].loc[closest_date]
                        div_return = div_amount / price_on_div if price_on_div > 0 else 0
                        total_returns.loc[closest_date] += div_return
                        
                        if ticker not in dividends_paid:
                            dividends_paid[ticker] = 0
                        dividends_paid[ticker] += div_amount
                
                total_returns_list.append(total_returns)
            except:
                # Fallback to price returns only
                price_returns = prices[ticker].pct_change().fillna(0)
                total_returns_list.append(price_returns)
        
        # Combine all returns
        returns_df = pd.concat(total_returns_list, axis=1, join='inner')
        returns_df.columns = available_tickers
        
        # Calculate portfolio returns
        portfolio_daily_returns = (returns_df * adjusted_weights).sum(axis=1)
        
        # Calculate portfolio value
        portfolio_value = initial_investment * (1 + portfolio_daily_returns).cumprod()
        
        # Current value and return
        current_value = portfolio_value.iloc[-1]
        total_return = ((current_value - initial_investment) / initial_investment) * 100
        
        # Calculate total dividends received
        total_dividends = sum([dividends_paid.get(t, 0) * adjusted_weights[i] * initial_investment / prices[t].iloc[0] 
                              for i, t in enumerate(available_tickers)])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Investment", f"${initial_investment:,.0f}")
        with col2:
            st.metric("Current Value", f"${current_value:,.2f}", f"{total_return:+.2f}%")
        with col3:
            gain_loss = current_value - initial_investment
            st.metric("Gain/Loss", f"${gain_loss:+,.2f}")
        with col4:
            st.metric("Dividends Received", f"${total_dividends:,.2f}", 
                     f"{len(dividends_paid)} stocks" if dividends_paid else "None")
        
        # Show which stocks paid dividends
        if dividends_paid:
            with st.expander("💰 Dividend Details"):
                div_details = []
                for ticker, div_amt in dividends_paid.items():
                    idx = available_tickers.index(ticker)
                    weight = adjusted_weights[idx]
                    initial_price = prices[ticker].iloc[0]
                    shares = (initial_investment * weight) / initial_price
                    total_div = div_amt * shares
                    div_details.append({
                        'Ticker': ticker,
                        'Dividend per Share': f"${div_amt:.4f}",
                        'Est. Shares': f"{shares:.2f}",
                        'Total Dividend': f"${total_div:.2f}"
                    })
                st.dataframe(pd.DataFrame(div_details), use_container_width=True)
        
        # Create tabs for graph and table
        tab1, tab2 = st.tabs(["📈 Graph", "📊 Table"])
        
        with tab1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Sharp Portfolio (incl. dividends)',
                line=dict(color='blue', width=2),
                fill='tonexty',
                fillcolor='rgba(0,100,255,0.1)'
            ))
            
            fig.add_hline(
                y=initial_investment,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Investment",
                annotation_position="right"
            )
            
            fig.update_layout(
                title="Sharp Portfolio Value Over Last Month (Including Dividends)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            performance_table = pd.DataFrame({
                'Date': portfolio_value.index.strftime('%Y-%m-%d'),
                'Portfolio Value ($)': portfolio_value.values.round(2),
                'Daily Return (%)': (portfolio_daily_returns * 100).values.round(2),
                'Total Return (%)': (((portfolio_value.values / initial_investment) - 1) * 100).round(2)
            })
            
            performance_table = performance_table.iloc[::-1].reset_index(drop=True)
            
            st.dataframe(
                performance_table.style.applymap(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                    subset=['Daily Return (%)', 'Total Return (%)']
                ),
                use_container_width=True,
                height=400
            )
            
            csv = performance_table.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"sharp_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error calculating 1-month performance: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")

# Calculate Sharpe Ratio
st.subheader("📊 Portfolio Analysis (5-Year Historical)")

with st.spinner("Calculating Sharpe Ratio (fetching 5 years of data)..."):
    try:
        result = calculate_sharpe_ratio(tickers, np.array(weights), risk_free_rate)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.4f}")
        with col2:
            st.metric("Annual Return", f"{result['annual_return']*100:.2f}%")
        with col3:
            st.metric("Annual Volatility", f"{result['annual_volatility']*100:.2f}%")
        with col4:
            st.metric("Risk-Free Rate", f"{risk_free_rate*100:.1f}%")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cumulative Returns (5 Years)")
            portfolio_returns = np.array(result['portfolio_returns'])
            cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                y=cumulative_returns * 100,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            fig1.update_layout(
                xaxis_title="Trading Days",
                yaxis_title="Returns (%)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Capital Market Line")
            
            annual_volatility = result['annual_volatility']
            annual_return = result['annual_return']
            sharpe_ratio = result['sharpe_ratio']
            
            x_range = np.linspace(0, annual_volatility * 100 * 1.5, 100)
            y_cml = risk_free_rate * 100 + sharpe_ratio * x_range
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=x_range,
                y=y_cml,
                mode='lines',
                name='CML',
                line=dict(color='blue', width=3)
            ))
            
            fig2.add_trace(go.Scatter(
                x=[0],
                y=[risk_free_rate * 100],
                mode='markers',
                name='Risk-Free Rate',
                marker=dict(size=15, color='green')
            ))
            
            fig2.add_trace(go.Scatter(
                x=[annual_volatility * 100],
                y=[annual_return * 100],
                mode='markers',
                name='Portfolio',
                marker=dict(size=15, color='red')
            ))
            
            fig2.update_layout(
                xaxis_title="Risk (Annual Volatility, %)",
                yaxis_title="Return (Annual, %)",
                hovermode='closest',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error calculating portfolio metrics: {str(e)}")

# Footer
st.markdown("---")
st.caption("Data from Yahoo Finance | Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
st.caption("⚠️ Note: Dividend data may be incomplete for some funds. Total return calculation is an estimate.")