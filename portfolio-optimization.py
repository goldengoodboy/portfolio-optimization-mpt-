# Modern  Portfolio Theory: Asset Allocation optimizaiton
# Asset: Gold(^XAU),S&P 500(^GSPC), BIT Coin (BTC-USD)

# Cell 1 import require library
import numpy  as np # numerical calculation
import pandas as pd # data manipulation
import yfinance as yf # import price data from  Yahoo Finance
import matplotlib.pyplot as plt # Visulaization
import seaborn as sns 
from scipy.optimize import minimize  # Portfolio optimization (minimum variance)
from datetime import datetime

# Set plot style 
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Cell 2: Define asset and download data form Yahoo Finance
tickers = ["^XAU","^GSPC","BTC-USD"]
start_date = "2015-01-01"
end_date = "2026-02-02"

print(f"Downloading data for {', '.join(tickers)}")
print(f"Period: {start_date} to {end_date}")

# Download adjusted close price
price_data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    interval="1mo",
    auto_adjust=True,
    progress=False
)["Close"]

print(f"\n✓ Download complete!")
print(f"  Raw data shape: {price_data.shape}")

# Cell 3: Clean data and log return calculation
print("\n" + "="*70)
print("DATA CLEANING")
print("="*70)

print("\nMissing values per ticker (before cleaning):")
missing_counts = price_data.isnull().sum()
print(missing_counts)

# Find the first date where ALL tickers have valid data
print("\nFinding first valid date for each ticker:")
first_valid_dates = {}
for ticker in tickers:
    first_valid = price_data[ticker].first_valid_index()
    first_valid_dates[ticker] = first_valid
    print(f"  {ticker:10s}: {first_valid}")

# Get the LATEST first valid date (when all assets are available)
latest_start = max(first_valid_dates.values())
print(f"\n✅ All tickers available from: {latest_start}")

# Slice data starting from when all assets are available
price_data_clean = price_data[price_data.index >= latest_start].copy()

print(f"✓ After aligning to common start date: {price_data_clean.shape[0]} observations")

# Double-check for any remaining missing values
remaining_missing = price_data_clean.isnull().sum()
if remaining_missing.sum() > 0:
    print(f"\n⚠️  Still have {remaining_missing.sum()} missing values:")
    print(remaining_missing)
    print("Dropping remaining rows with missing values...")
    price_data_clean = price_data_clean.dropna()
    print(f"✓ After final cleaning: {price_data_clean.shape[0]} observations")

# Verify we have enough data
if len(price_data_clean) < 24:
    raise ValueError(f"❌ Insufficient data! Only {len(price_data_clean)} months. Need at least 24.")

print(f"\n✓ Final cleaned data: {price_data_clean.shape[0]} observations")
print(f"  Date range: {price_data_clean.index[0]} to {price_data_clean.index[-1]}")

# Calculate log returns
log_returns = np.log(price_data_clean / price_data_clean.shift(1)).dropna()

print(f"\n✓ Log returns calculated: {log_returns.shape[0]} observations")
print(f"  Date range: {log_returns.index[0]} to {log_returns.index[-1]}")

# Preview
print("\nFirst 5 rows of log returns:")
print(log_returns.head())

print("\nLast 5 rows of log returns:")
print(log_returns.tail())

# Annualize returns and volatility (12 months per year)
mean_returns_annual = log_returns.mean() * 12
cov_matrix_annual = log_returns.cov() * 12

print("\n" + "="*70)
print("ANNUALIZED STATISTICS")
print("="*70)

print("\nAnnualized Expected Returns:")
for asset, ret in mean_returns_annual.items():
    print(f"  {asset:10s}: {ret*100:6.2f}%")
    
print("\nAnnualized Volatility (Std Dev):")
for asset, vol in (log_returns.std() * np.sqrt(12)).items():
    print(f"  {asset:10s}: {vol*100:6.2f}%")

print("\nCorrelation Matrix:")
print(log_returns.corr().round(3))

# Verify covariance matrix is valid
print("\n" + "="*70)
print("Covariance Matrix Validation:")
try:
    np.linalg.cholesky(cov_matrix_annual)
    print("✓ Covariance matrix is positive definite (VALID for optimization)")
except np.linalg.LinAlgError:
    print("❌ WARNING: Covariance matrix is NOT positive definite!")
    print("   Optimization may fail!")

print("="*70)

# DEBUG: Check data quality
print("\n" + "="*70)
print("DATA QUALITY CHECK")
print("="*70)
print(f"\n1. Data shape: {log_returns.shape}")
print(f"2. Columns: {list(log_returns.columns)}")
print(f"3. Date range: {log_returns.index[0]} to {log_returns.index[-1]}")

print("\n4. Missing/Invalid values:")
print(log_returns.isnull().sum())

print("\n5. Infinite values:")
print(np.isinf(log_returns).sum())

print("\n6. Annualized mean returns:")
print(mean_returns_annual)

print("\n7. Annualized volatilities:")
print(log_returns.std() * np.sqrt(12))

print("\n8. Covariance matrix:")
print(cov_matrix_annual)

print("\n9. Covariance matrix diagonal (variances):")
print(np.diag(cov_matrix_annual))

# Check if matrix is valid
try:
    np.linalg.cholesky(cov_matrix_annual)
    print("\n✓ Covariance matrix is positive definite (VALID)")
except:
    print("\n❌ PROBLEM: Covariance matrix is NOT positive definite!")
    print("   This will cause all optimizations to fail!")

print("="*70)

# Cell 4: Portfolio Metrix Functions
def portfolio_return(weights, returns):
    """Calculate ortffolio expected return"""
    return np.dot(weights, returns)

def portfolio_volatility(weights, cov_matrix):
    """Calculate portfolio volatility (risk)"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02):
    """Calculate Sharpe ratio (risk-adjusted return)"""
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol if vol > 0 else 0

# Cell 5: Equal Weight Porffolio Benchmark
n_assets = len(mean_returns_annual)
weights_equal = np.array([1/n_assets] * n_assets)

eq_return = portfolio_return(weights_equal, mean_returns_annual)
eq_vol = portfolio_volatility(weights_equal, cov_matrix_annual)
eq_sharpe = portfolio_sharpe_ratio(weights_equal, mean_returns_annual, cov_matrix_annual)

print("\n=== Equal Weight Portfolio (Benchmark) ===")
print(f"  Weights: {dict(zip(price_data.columns, np.round(weights_equal*100, 1)))}")
print(f"  Expected Return: {eq_return*100:.2f}%")
print(f"  Volatility: {eq_vol*100:.2f}%")
print(f"  Sharpe Ratio: {eq_sharpe:.3f}")

# Cell 6: Global Minimum Variance (GMV) Portfolio 
# Constraints: Weight sum to 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
# Bounds : NO short selling (0 <= weights <= 1 )
bounds = tuple((0,1) for _ in range(n_assets))
# Initial guess: Equal weights
initial_weights = weights_equal

# Minimize Volatility
opt_result = minimize(
    portfolio_volatility,
    initial_weights,
    args=(cov_matrix_annual,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Optimize Weight
weights_gmv = opt_result.x
#GMV: Global Minimum Variance Portfolio > portfolio return and risk
gmv_return = portfolio_return(weights_gmv, mean_returns_annual)
gmv_vol  = portfolio_volatility(weights_gmv, cov_matrix_annual)
gmv_sharpe = portfolio_sharpe_ratio(weights_gmv, mean_returns_annual, cov_matrix_annual)

print("\n=== Global Minimum Variance  (GMV) Portfolio ===")
print(f"Weights:{dict(zip(price_data.columns, np.round(weights_gmv*100, 1)))}")
print(f"Expected Return: {gmv_return*100:.2f}%")
print(f"Volatility: {gmv_vol*100:.2f}%")
print(f"Sharpe Ratio: {gmv_sharpe:.3f}")

# Cell 7: Efficient Frontier Construction
target_returns = np.linspace(mean_returns_annual.min(), mean_returns_annual.max(), 50)
frontier_vol = []
frontier_weights = []

for target_return in target_returns:
    # Constraints: Weight sum to 1 and target return
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, target_return: portfolio_return(w, mean_returns_annual) - target_return, 'args': (target_return,)}
    )
    
    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(cov_matrix_annual,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        frontier_vol.append(result.fun)
        frontier_weights.append(result.x)
    else:
        frontier_vol.append(np.nan)
        frontier_weights.append(None)

# Convert to numpy array for easier handling
frontier_vol = np.array(frontier_vol)

# Cell 8: Max Sharpe Ratio Portfolio (Tangency Portfolio)
def negative_sharpe(weights, returns, cov_matrix, risk_free_rate):
    return -portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)

opt_sharpe = minimize(
    negative_sharpe,
    initial_weights,
    args=(mean_returns_annual, cov_matrix_annual, 0.02),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

weights_max_sharpe = opt_sharpe.x
max_sharpe_return = portfolio_return(weights_max_sharpe, mean_returns_annual)
max_sharpe_vol = portfolio_volatility(weights_max_sharpe, cov_matrix_annual)
max_sharpe_ratio = portfolio_sharpe_ratio(weights_max_sharpe, mean_returns_annual, cov_matrix_annual, 0.02)

print("\n=== Maximum Sharpe Ratio Portfolio (Optimal Risk-Adjusted) ===")
print(f"  Weights: {dict(zip(price_data.columns, np.round(weights_max_sharpe*100, 1)))}")
print(f"  Expected Return: {max_sharpe_return*100:.2f}%")
print(f"  Volatility:      {max_sharpe_vol*100:.2f}%")
print(f"  Sharpe Ratio:    {max_sharpe_ratio:.3f}")

# CELL 9: Professional Visualization

plt.figure(figsize=(12, 8))

# Plot efficient frontier (only if we have valid data)
valid_idx = ~np.isnan(frontier_vol)

if np.sum(valid_idx) > 0:
    plt.plot(
        frontier_vol[valid_idx] * 100, 
        target_returns[valid_idx] * 100,
        color='darkblue', linewidth=2.5, label='Efficient Frontier'
    )
    print("✓ Efficient frontier plotted successfully")
else:
    print("⚠️  Warning: Efficient frontier optimization failed for all target returns")
    print("   Plotting portfolios only (without frontier curve)")

# Plot individual assets - FIX THE INDEX ACCESS
asset_names = list(price_data.columns)
asset_vols = log_returns.std() * np.sqrt(12)
asset_returns = mean_returns_annual

for i, asset in enumerate(asset_names):
    # ✅ Use .iloc for position-based indexing
    plt.scatter(
        asset_vols.iloc[i] * 100,
        asset_returns.iloc[i] * 100,
        s=150, alpha=0.8, edgecolors='black', linewidth=2,
        label=f'{asset} (Individual Asset)'
    )

# Plot portfolios - CHECK FOR NaN VALUES
if not np.isnan(eq_vol) and not np.isnan(eq_return):
    plt.scatter(eq_vol*100, eq_return*100, 
                color='green', s=250, marker='o', 
                edgecolors='black', linewidth=2, label='Equal Weight')

if not np.isnan(gmv_vol) and not np.isnan(gmv_return):
    plt.scatter(gmv_vol*100, gmv_return*100, 
                color='red', s=300, marker='*', 
                edgecolors='black', linewidth=2, label='GMV Portfolio')

if not np.isnan(max_sharpe_vol) and not np.isnan(max_sharpe_return):
    plt.scatter(max_sharpe_vol*100, max_sharpe_return*100, 
                color='gold', s=300, marker='D', 
                edgecolors='black', linewidth=2, label='Max Sharpe Portfolio')

# Add risk-free rate line (Capital Market Line)
if not np.isnan(max_sharpe_vol) and max_sharpe_vol > 0:
    risk_free_rate = 0.02
    cml_x = np.array([0, max_sharpe_vol*1.5]) * 100
    cml_y = risk_free_rate * 100 + (max_sharpe_ratio * cml_x)
    plt.plot(cml_x, cml_y, 'r--', linewidth=1.5, alpha=0.7, label='Capital Market Line')

# Formatting
plt.title('Modern Portfolio Theory: Efficient Frontier & Optimal Portfolios\n'
          f'Assets: {", ".join(asset_names)} | Period: {start_date} to {end_date}', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Annualized Volatility (Risk) [%]', fontsize=12, fontweight='bold')
plt.ylabel('Annualized Expected Return [%]', fontsize=12, fontweight='bold')
plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
plt.grid(True, alpha=0.3)

# ✅ FIX: Set axis limits safely
all_vols = []
all_returns = []

# Collect valid volatilities
if np.sum(valid_idx) > 0:
    all_vols.extend(frontier_vol[valid_idx] * 100)
for vol in [eq_vol, gmv_vol, max_sharpe_vol]:
    if not np.isnan(vol) and vol > 0:
        all_vols.append(vol * 100)
# Add asset volatilities
all_vols.extend(asset_vols * 100)

# Collect valid returns
if np.sum(valid_idx) > 0:
    all_returns.extend(target_returns[valid_idx] * 100)
for ret in [eq_return, gmv_return, max_sharpe_return]:
    if not np.isnan(ret):
        all_returns.append(ret * 100)
# Add asset returns
all_returns.extend(asset_returns * 100)

# Set limits
if all_vols:
    plt.xlim(0, max(all_vols) * 1.1)
else:
    plt.xlim(0, 50)

if all_returns:
    plt.ylim(min(all_returns) * 0.9, max(all_returns) * 1.1)
else:
    plt.ylim(-10, 30)

# Add annotations (only if portfolios exist)
if not np.isnan(gmv_vol) and not np.isnan(gmv_return):
    plt.annotate('GMV\n(Lowest Risk)', 
                 xy=(gmv_vol*100, gmv_return*100), 
                 xytext=(gmv_vol*100+2, gmv_return*100-3),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=9, color='red', fontweight='bold')

if not np.isnan(max_sharpe_vol) and not np.isnan(max_sharpe_return):
    plt.annotate('Max Sharpe\n(Optimal)', 
                 xy=(max_sharpe_vol*100, max_sharpe_return*100), 
                 xytext=(max_sharpe_vol*100+2, max_sharpe_return*100+2),
                 arrowprops=dict(arrowstyle='->', color='gold', lw=1.5),
                 fontsize=9, color='darkorange', fontweight='bold')

plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved as 'efficient_frontier.png'")
plt.show()

# CELL 9: Portfolio Comparison Table
portfolio_comparison = pd.DataFrame({
    'Portfolio': ['Equal Weight', 'GMV', 'Max Sharpe'],
    'Gold (%)': [
        weights_equal[0]*100,
        weights_gmv[0]*100,
        weights_max_sharpe[0]*100
    ],
    'S&P 500 (%)': [ 
        weights_equal[1]*100,
        weights_gmv[1]*100, 
        weights_max_sharpe[1]*100
    ],
    'Bitcoin (%)': [
        weights_equal[2]*100,
        weights_gmv[2]*100,
        weights_max_sharpe[2]*100
    ],
    'Expected Return (%)': [
        eq_return*100,
        gmv_return*100,
        max_sharpe_return*100
    ],
    'Volatility (%)': [
        eq_vol*100,
        gmv_vol*100,
        max_sharpe_vol*100
    ],
    'Sharpe Ratio': [
        eq_sharpe,
        gmv_sharpe,
        max_sharpe_ratio
    ]
})

print("\n=== PORTFOLIO COMPARISON TABLE ===")
print(portfolio_comparison.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

