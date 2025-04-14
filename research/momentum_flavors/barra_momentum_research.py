import os
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

import silverfund.data_access_layer_v2 as dal

os.makedirs("research/momentum_flavors/results/barra", exist_ok=True)


def create_portfolios(signals: pl.DataFrame, signal: str) -> pl.DataFrame:
    # ----- Create Decile Portfolios -----
    labels = [str(i) for i in range(10)]
    return (
        signals
        # Decile cut
        .with_columns(pl.col(signal).qcut(10, labels=labels).over("date").alias("bin"))
        # Equal weight portfolios
        .group_by(["date", "bin"]).agg(pl.col("return").mean())
        # Bins wide
        .pivot(index="date", on="bin", values="return")
        # Spread portfolios
        .with_columns(pl.col("9").sub(pl.col("0")).alias("spread"))
    )


def create_cummulative_returns(portfolios: pl.DataFrame) -> pl.DataFrame:
    # ----- Create Cummulative Returns -----
    return (
        portfolios
        # Bins long
        .unpivot(index="date", variable_name="bin", value_name="return")
        .sort(["date", "bin"])
        # Cummulative log return
        .with_columns(pl.col("return").truediv(100).log1p().cum_sum().mul(100).over("bin").alias("cummulative_return"))
        # Bins wide
        .pivot(index="date", on="bin", values="cummulative_return")
        .sort("date")
    )


def create_backtest_chart(portfolios: pl.DataFrame, cummulative_returns: pl.DataFrame, signal: str) -> None:
    signal_title = signal.replace("_", " ").title()
    # ----- Create Chart -----
    sharpe = portfolios["spread"].mean() / portfolios["spread"].std() * np.sqrt(252)

    plt.figure(figsize=(10, 6))

    bins = list(range(10))
    colors = sns.color_palette("coolwarm", len(bins))

    for i, color in zip(bins, colors):
        if i in [0, 9]:
            bin = str(i)
            sns.lineplot(cummulative_returns, x="date", y=bin, color=color, label=bin)

    sns.lineplot(cummulative_returns, x="date", y="spread", color="green", label="spread")

    plt.title(f"{signal_title} ({sharpe:.2f})")
    plt.xlabel(None)
    plt.ylabel("Cummulative Sum Return (%)")
    plt.legend(title="Portfolio")

    plt.savefig(f"research/momentum_flavors/results/barra/barra_{signal}.png", dpi=300)
    plt.clf()


# ----- Parameters -----
start_date = date(1995, 1, 1)
end_date = date(2024, 12, 31)

# ----- Load Data -----
data = dal.assets.load(
    start_date=start_date,
    end_date=end_date,
    in_universe=True,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
        "return",
        "specific_return",
        "russell_1000",
        "russell_2000",
        "market_cap",
    ],
)

print("Data", data)

# ----- Compute Signals -----
signals = (
    data
    # Sort
    .sort(["barrid", "date"])
    .with_columns(
        # Vanilla Momentum
        pl.col("return").truediv(100).log1p().rolling_sum(window_size=230).over("barrid").alias("momentum"),
        # Idiosyncratic Momentum
        pl.col("specific_return").truediv(100).log1p().rolling_sum(window_size=230).over("barrid").alias("idiosyncratic_momentum"),
        # Volatility
        pl.col("return").truediv(100).rolling_std(window_size=230).over("barrid").alias("volatility"),
    )
    # Lag
    .with_columns(
        pl.col("momentum").shift(22).over("barrid"),
        pl.col("idiosyncratic_momentum").shift(22).over("barrid"),
        pl.col("volatility").shift(22).over("barrid"),
    )
    # Volatility adjusted momentum
    .with_columns(pl.col("momentum").truediv(pl.col("volatility")).alias("volatility_adjusted_momentum"))
    # Percent up and percent down days
    .with_columns(
        pl.col("return").gt(0).cast(pl.Int32).alias("positive"),
        pl.col("return").lt(0).cast(pl.Int32).alias("negative"),
    )
    .sort(["barrid", "date"])
    .with_columns(
        pl.col("positive").rolling_sum(window_size=22).over("barrid"),
        pl.col("negative").rolling_sum(window_size=22).over("barrid"),
    )
    # Information discreteness
    .with_columns(pl.col("momentum").sign().mul(pl.col("negative").sub(pl.col("positive")).truediv(22)).alias("id"))
    .with_columns(pl.col("id").shift(1).over("barrid"))
    # Frog in the pan signal
    .with_columns(pl.col("momentum").sub(pl.col("momentum").mean()).truediv(pl.col("momentum").std()).over("date").alias("z_momentum"))
    .with_columns(pl.col("id").sub(pl.col("id").mean()).truediv(pl.col("id").std()).over("date").alias("z_id"))
    .with_columns(pl.col("z_momentum").mul(pl.col("z_id")).alias("frog_in_the_pan_momentum"))
    # Rename vanilla momentum
    .with_columns(pl.col("momentum").alias("vanilla_momentum"))
    # Price and null filter
    .with_columns(pl.col("price").shift(1).over("barrid").alias("price_lag"))
)

print("Signals", signals)
print(signals.group_by("date").agg(pl.len()).sort("date"))

signals = (
    signals.filter(pl.col("price_lag").gt(5))
    .drop_nulls(
        [
            "vanilla_momentum",
            "volatility_adjusted_momentum",
            "frog_in_the_pan_momentum",
            "idiosyncratic_momentum",
        ]
    )
    .sort(["barrid", "date"])
)

print("Signals", signals)
print(signals.group_by("date").agg(pl.len()).sort("date"))

signal_names = [
    "vanilla_momentum",
    "volatility_adjusted_momentum",
    "frog_in_the_pan_momentum",
    "idiosyncratic_momentum",
]

combined = None
sharpes = {}
for signal_name in signal_names:
    portfolios = create_portfolios(signals, signal_name)

    sharpe = portfolios["spread"].mean() / portfolios["spread"].std() * np.sqrt(252)
    sharpes[signal_name] = sharpe
    cummulative_returns = create_cummulative_returns(portfolios)

    if combined is None:
        combined = cummulative_returns.select("date", pl.col("spread").alias(signal_name))

    else:
        combined = combined.join(
            cummulative_returns.select("date", pl.col("spread").alias(signal_name)),
            on="date",
            how="left",
        )

    create_backtest_chart(portfolios, cummulative_returns, signal_name)


# ----- Combined Plot -----
plt.figure(figsize=(10, 6))

for signal_name in signal_names:
    signal_title = " ".join(signal_name.split("_")[:-1]).title()
    label = f"{signal_title} ({sharpes[signal_name]:.2f})"

    sns.lineplot(combined, x="date", y=signal_name, label=label)

plt.title("Momentum Strategies")
plt.xlabel(None)
plt.ylabel("Cummulative Sum Return (%)")
plt.legend(title="Strategy (Sharpe)")

plt.savefig("research/momentum_flavors/results/barra/barra_combined_plot.png", dpi=300)
