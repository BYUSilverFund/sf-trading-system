"""Data Access Layer (DAL) for financial and market data.

This module provides functions to load various financial datasets, including
Barra factor models, CRSP data, trading days, and benchmark data. The functions
facilitate efficient data retrieval and preprocessing for quantitative analysis.

Available functions:
- load_trading_days: Retrieves trading days based on a given interval.
- load_universe: Loads the stock universe with Russell constituents.
- load_total_risk: Retrieves total risk estimates from Barra.
- load_barra_returns: Loads factor return data.
- load_crsp: Retrieves CRSP stock market data.
- load_specific_returns: Loads specific return data from Barra.
- load_factor_covariances: Retrieves factor covariance matrices.
- load_factor_exposures: Loads factor exposure data.
- load_specific_risk: Retrieves specific risk estimates from Barra.
- load_benchmark: Loads benchmark return data.

These functions help streamline access to structured market and risk model data.
"""

from . import assets, crsp_daily, crsp_monthly

__all__ = ["assets", "crsp_daily", "crsp_monthly"]
