import pandas as pd

from silverfund.backtester import MVBacktester
from silverfund.components.optimizers.constraints import *

# %%
alphas_df = pd.read_parquet("/home/bwaits/Research/sf-trading-system/reversal_research/alphas.parquet")
# %%
mv_backtester = MVBacktester(alpha=alphas_df.copy(), constraints=[FullInvestmentConstraint()])
# %%
port = mv_backtester.get_optimal_portfolio_history(n_cpus=16)
# %%
port.to_parquet("/home/bwaits/Research/sf-trading-system/reversal_research/portfolios.parquet")
