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
- Libraries: `numpy`, `pandas`, `cvxpy`, `scipy`, `scikit-learn`, `matplotlib`  
- Optional: `networkx` for clustering, `yfinance` for data ingestion  

Install via:
```bash
pip install -r requirements.txt
```

## 🛠️ Structure

```
.
├── data/               # Price/returns datasets
├── optimization/       # Core modules (estimators, optimizers, constraints)
├── strategies/         # Workflow scripts and notebook demos
├── tests/              # Unit tests (pytest compatible)
├── notebooks/          # Jupyter notebooks illustrating use cases
└── README.md
```

## 🚀 Quickstart

```python
import pandas as pd
from optimization.expected_returns import mean_historical_return
from optimization.risk_models import sample_covariance
from optimization.optimizers import EfficientFrontier

prices = pd.read_csv('data/prices.csv', index_col='Date', parse_dates=True)
mu = mean_historical_return(prices)
Sigma = sample_covariance(prices)

ef = EfficientFrontier(mu, Sigma)
ef.add_l2_regularization(gamma=1.0)
weights = ef.max_sharpe()
print(weights)
```

## 📊 Backtesting Example

In `strategies/backtest_portfolio.py`, a walk‑forward simulation is performed:

1. Estimate mu and Sigma over rolling window  
2. Optimize portfolio  
3. Track performance out-of-sample  

Use `pytest` to verify functionality and `matplotlib` to plot returns curves.

## 🧩 Extend & Customize

- Add new covariance estimators (e.g. robust, factor-based)  
- Implement alternative objective functions (e.g. CVaR, utility)  
- Introduce portfolio-level constraints (sector, ESG, turnover)  
- Swap linear/quadratic solvers in `cvxpy`

## ✅ Testing & Continuous Integration

- Extensive unit coverage in `tests/` using `pytest`  
- Run all tests:
  ```bash
  pytest
  ```
- CI pipelines configured for test validation on each commit

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
