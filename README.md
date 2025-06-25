# Portfolio Optimization

This project implements advanced portfolio optimization techniques in Python, based on Markowitz, Black–Litterman, Hierarchical Risk Parity, and more. It provides tools to construct portfolios that balance risk and return, tailored for both practitioners and researchers.

## 🔍 Features

- **Mean–Variance Optimization**: Efficient frontier, minimum variance, maximum Sharpe ratio  
- **Black–Litterman Allocation**: Integrate investor views with market equilibrium returns  
- **Hierarchical Risk Parity**: Cluster-based risk balancing approach  
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
- Optional: `networkx` for clustering, `yfinance` for data ingestion  

Install via:
```bash
pip install -r requirements.txt
```

## 🛠️ Structure

```
.
├── data/                           # multifactor fama french data
├── exports/                        # exports CSVs and PDFs of results
├── port_details.txt                # user details for portfolio optimization notebook
├── portfolio-optimization.ipynb    # notebook containing code for port optim.
└── README.md
```

## 🚀 Quickstart

```python
import pandas as pd
from optimization.expected_returns import mean_historical_return
from optimization.risk_models import sample_covariance
from optimization.optimizers import EfficientFrontier

# will need to update directory to reflect where you are getting data
prices = pd.read_csv('data/prices.csv', index_col='Date', parse_dates=True)
mu = mean_historical_return(prices)
Sigma = sample_covariance(prices)

ef = EfficientFrontier(mu, Sigma)
ef.add_l2_regularization(gamma=1.0)
weights = ef.max_sharpe()
print(weights)
```


## 🧩 Extend & Customize

- Add new covariance estimators (e.g. robust, factor-based)  
- Implement alternative objective functions (e.g. CVaR, utility)  
- Introduce portfolio-level constraints (sector, ESG, turnover)  
- Swap linear/quadratic solvers in `cvxpy`


## 📚 References

- H. Markowitz (1952), *Portfolio Selection*  
- He, E., & Litterman (1999), *The Intuition Behind Black–Litterman*  
- López de Prado (2016), *Hierarchical Risk Parity*  
- Academic and practical finance literature

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
