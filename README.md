# Portfolio Optimization

This project implements portfolio optimization using Monte Carlo simulation and Sharpe ratio maximization. It allows users to define custom constraints, risk-free rates, and asset classes using historical ETF data. Ideal for analysts and researchers aiming to simulate portfolio performance over a 10-year horizon.

## 🔍 Features

- **Mean–Variance Optimization**: Efficient frontier, minimum variance, maximum Sharpe ratio  
- **Covariance Estimators**: Sample, shrinkage, exponential, semi-covariance  
- **Regularization**: Enforce sparsity through L1/L2 objectives  
- **Custom Constraints**: Short/long bounds, turnover/weight limits, market neutrality, sector exposure  
- **Backtesting Support**: Plug into historical price data workflows  
- **Modular Design**: Swap models with minimal code changes

## ⚙️ Requirements

- Python 3.8+  
- Libraries:
  - `yfinance`
  - `pandas`
  - `pandas_datareader`
  - `matplotlib`
  - `numpy`
  - `seaborn`
  - `arch`
  - `plotly`
  - `statsmodels`
  - `nbformat`
  - `nbconvert`
  - `ipykernel`
  - `ipywidgets`
  - `scipy`
  - `kaleido`

Install all dependencies via:

```bash
pip install -r requirements.txt
```

To set up your environment:

1. Clone the repository.
2. Create a virtual environment (optional but recommended).
3. Run the above install command.

---

### 🔧 User Customization Instructions

Before running the notebook:

- **Tickers**:  
  In `portfolio-optimization.ipynb`, update the `tickers` list to match the assets you wish to analyze. These will be used to download pricing data and calculate expected returns and risk.

- **Constraints**:  
  Modify or extend the constraints section (e.g. weight bounds, sector exposures, turnover limits) to suit your portfolio rules. Most constraints are passed directly into the optimizer methods.

- **Risk-Free Rate**:  
  Set the appropriate risk-free rate for Sharpe ratio calculations. This is typically done in a variable such as:

  ```python
  risk_free_rate = 0.03  # update based on your region or target benchmark
  ```

  Ensure this value is consistent across all relevant calculations.

## 🛠️ Structure

```
.
├── exports/                        # exports CSVs and PDFs of results
├── ff_factor_data/                 # multifactor fama french data
├── port_details.txt                # user details for portfolio optimization notebook
├── portfolio-optimization.ipynb    # notebook containing code for port optim.
└── README.md
```

## 🚀 Quickstart

```python
import numpy as np
import pandas as pd

# Load prices — update the path if needed
prices = pd.read_csv('exports/prices_weekly_10yrs.csv', index_col='Date', parse_dates=True)

# Calculate weekly returns
returns = prices.pct_change().dropna()

# Expected returns: mean historical return
mu = returns.mean()

# Covariance matrix
Sigma = returns.cov()

# Monte Carlo simulation to generate random portfolios
n_portfolios = 50000
n_assets = len(mu)
results = np.zeros((3, n_portfolios))
weights_record = []

risk_free_rate = 0.03 / 52  # weekly risk-free rate

for i in range(n_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)
    weights_record.append(weights)
    
    port_return = np.dot(weights, mu)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    
    results[0, i] = port_return
    results[1, i] = port_volatility
    results[2, i] = sharpe_ratio

# Find the portfolio with the max Sharpe ratio
max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]
print("Optimal Weights:", dict(zip(prices.columns, optimal_weights)))
```


## 🧩 Extend & Customize

- Add new covariance estimators (e.g. robust, factor-based)  
- Implement alternative objective functions (e.g. CVaR, utility)  
- Introduce portfolio-level constraints (sector, ESG, turnover)  
- Swap linear/quadratic solvers in `cvxpy`
- Incorporate investor views into the portfolio optimization process. (Litterman)



## 📚 References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, 7(1), 77–91.  
- Black, F., & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal, 48(5), 28–43.  
- López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out of Sample*. The Journal of Portfolio Management, 42(4), 59–69.  
- Fama, E.F., & French, K.R. (1993). *Common Risk Factors in the Returns on Stocks and Bonds*. Journal of Financial Economics, 33(1), 3–56.  
  - Data retrieved from the Kenneth R. French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html  
- Yahoo Finance via `yfinance` Python package  
  - Ran Aroussi (2020). *yfinance – Yahoo! Finance market data downloader*. GitHub repository: https://github.com/ranaroussi/yfinance  
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.  
- Diamond, S., & Boyd, S. (2016). *CVXPY: A Python-Embedded Modeling Language for Convex Optimization*. Journal of Machine Learning Research, 17(83), 1–5.  
- Fabozzi, F.J., Kolm, P.N., Pachamanova, D.A., & Focardi, S.M. (2007). *Robust Portfolio Optimization and Management*. Wiley Finance.  
- Grinold, R.C., & Kahn, R.N. (2000). *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk*. McGraw-Hill.


## 🏗️ Roadmap

- Add CVaR, drawdown-based objectives  
- Support multi-period, dynamic rebalancing  
- Integrate factor models and robust covariance estimation  
- Deploy plug-and-play with live data feeds

## 🙏 Contributing

- Fork → add feature or fix → open PR  
- All enhancements, bug fixes, and documentation welcomed  
- Please ensure new features have tests and conform to code style

## 📄 License

Licensed under the **MIT License**. Free to use, modify, and distribute.
