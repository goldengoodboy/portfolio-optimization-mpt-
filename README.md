# Modern Portfolio Theory: Portfolio Optimization
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-green?logo=matplotlib)
![yfinance](https://img.shields.io/badge/yfinance-0.2%2B-black?logo=yahoo)

## 🔍 Project Overview
Implemented **Harry Markowitz's Modern Portfolio Theory (1952)** to optimize asset allocation across three low-correlation asset classes using 11+ years of historical data (2015-2026):
- **Gold (^XAU)**: Global safe-haven asset.
- **S&P 500 (^GSPC)**: Broad US equity market benchmark representing growth-oriented capital allocation
- **Bitcoin (BTC-USD)**: Digital asset, crypto currency.

Key insight: **Gold's minimal correlation with Bitcoin (0.130)** creates powerful diversification benefits—critical for institutional portfolios seeking crisis resilience, directly relevant to Thailand's foreign reserve management framework.

## 📊 Key Results
![Efficient Frontier](efficient_frontier.png)

| Portfolio | Gold (%) | S&P 500 (%) | Bitcoin (%) | Return | Volatility | Sharpe Ratio |
|-----------|----------|-------------|-------------|--------|------------|--------------|
| **Equal Weight** | 33.3 | 33.3 | 33.3 | 26.17% | 29.07% | **0.83** |
| **GMV** (Lowest Risk) | 4.8 | 95.2 | 0.0 | 11.34% | 14.80% | 0.63 |
| **Max Sharpe** (Unconstrained) | 0.0 | 0.0 | 100.0 | 52.80% | 66.68% | 0.76 |

### 💡 Critical Professional Observations
1. **Diversification Premium**: The Equal Weight portfolio achieves a **higher Sharpe ratio (0.83) than the unconstrained Max Sharpe portfolio (0.76)**—demonstrating that naive diversification can outperform single-asset concentration when assets exhibit low correlations (Bitcoin-Gold: 0.130).

2. **GMV Portfolio Composition**: 95.2% allocation to S&P 500 reflects its status as the **lowest-volatility asset** (14.89% annualized) in this universe—counterintuitive given Gold's "safe-haven" reputation but mathematically sound based on historical volatility.

3. **Institutional Constraint Requirement**: While unconstrained optimization suggests 100% Bitcoin allocation, Thailand's SEC Regulation 29/2564 limits digital asset exposure to **30% of institutional portfolios**. This demonstrates why real-world portfolio construction requires explicit allocation constraints.

## 🌏 Relevance to Thai Financial Institutions
| Institution | Application Insight |
|-------------|---------------------|
| **Bank of Thailand** | Gold's low correlation with risk assets (0.130 with Bitcoin, 0.309 with S&P 500) provides crisis hedge properties valuable for foreign reserve diversification—critical lesson from 1997 Asian Financial Crisis |
| **NESDB** | Demonstrates evidence-based asset allocation methodology for Thailand's national development funds under volatile global conditions |
| **SEC Thailand** | Highlights necessity of regulatory constraints (e.g., 30% crypto cap) to prevent concentration risk in institutional portfolios |

## 🛠️ Technical Implementation
| Component | Implementation Details |
|-----------|------------------------|
| **Data Source** | Yahoo Finance API (`yfinance`), monthly adjusted close prices (Jan 2015 – Feb 2026) |
| **Asset Selection Rationale** | Chosen for low pairwise correlations (max 0.355 between Bitcoin/S&P 500) to maximize diversification benefits |
| **Optimization Engine** | SciPy `minimize()` with SLSQP method, no-short-selling constraints (`0 ≤ weights ≤ 1`) |
| **Risk Metrics** | Annualized volatility (×√12), Sharpe ratio (risk-free rate = 2% annual) |
| **Validation** | Covariance matrix confirmed positive definite (eigenvalues > 0) ensuring mathematical validity |

## 📁 Repository Structure
portfolio-optimization-mpt/        
├── portfolio_optimization.py     
├── efficient_frontier.png      
├── requirements.txt             
└── README.md   
