#   Module: barra_data_assembly.py
#   Author: Brandon Bates and Seth Peterson
#   Date: October 2023
#   Purpose: retrieve barra model components
# ------------------------------------------------------------------------------------------------ #
# Contents:
#   - load_exposure_matrix
#   - load_factor_covariance_matrix
#   - load_specific_covariance_matrix
#   - load_r3k_benchmark_weights
#   - cusip_to_barra_id
#
# ------------------------------------------------------------------------------------------------ #


# --- Import Modules ---
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --- Options and Parameters ---

# ------------------------------------------------------------------------------------------------ #
#                                    Barra Risk Model Component Loaders
# ------------------------------------------------------------------------------------------------ #


def load_exposure_matrix(
    date: str | pd.Timestamp,
    barrids: list[str] | pd.Index = None,
    factorids: list[str] | pd.Index = None,
    omitted_factors: list[str] | pd.Index = None,
) -> pd.DataFrame:
    """Load the exposure matrix from the consolidated barra parquet files.

    Parameters
    ----------
    date: str | pd.Timestamp
        '2010-03-04' or '20100304' format or a pd.Timestamp
    barrids: list[str], optional
        list of barra ids
    factorids: list[str], optional
        list of factor ids to keep in the output

    Returns
    -------
    pd.DataFrame
        dimension N x k
    """
    ix_slc = pd.IndexSlice

    # extract exposure matrix on one date
    pq_date = pd.to_datetime(date)
    pq_fname = FACTOR_EXPOSURE_FILES[pq_date.year]

    raw_exp_mat = pd.read_parquet(pq_fname, columns=[pq_date.__str__()])

    raw_exp_mat.index = raw_exp_mat.index.str.split("/", expand=True)
    raw_exp_mat.index.names = ["barrid", "factorid"]

    all_factor_ids = raw_exp_mat.index.get_level_values(level=1).unique()

    if factorids is not None:
        factorids = [fid.upper() for fid in factorids]
        all_factor_ids = all_factor_ids.intersection(factorids)

    # extract relevant barrids and factorids
    exp_mat = raw_exp_mat.loc[ix_slc[barrids, :], :] if barrids is not None else raw_exp_mat
    exp_mat = exp_mat.loc[ix_slc[:, factorids], :] if factorids is not None else exp_mat
    pivoted_exp_mat = exp_mat.reset_index().pivot(
        index="barrid", columns="factorid", values=exp_mat.columns.values
    )
    pivoted_exp_mat = pivoted_exp_mat.droplevel(axis=1, level=0)
    pivoted_exp_mat[all_factor_ids.difference(pivoted_exp_mat.columns).to_list()] = np.nan
    pivoted_exp_mat.replace(np.nan, 0, inplace=True)

    # Remove omitted factors
    if omitted_factors is not None:
        omitted_factors_upper = [x.upper() for x in omitted_factors]
        pivoted_exp_mat = pivoted_exp_mat.drop(omitted_factors_upper, axis=1)

    return pivoted_exp_mat.sort_index(axis=0).sort_index(axis=1)


def load_factor_covariance_matrix(
    date: str | pd.Timestamp,
    factorids: list[str] | pd.Index = None,
    omitted_factors: list[str] = None,
) -> pd.DataFrame:
    """Load the factor covariance matrix from the consolidated barra parquet files.

    Parameters
    ----------
    date: str | pd.Timestamp
        '2010-03-04' or '20100304' format or a pd.Timestamp
    factorids: list[str], optional
        list of factor ids to keep in the output

    Returns
    -------
    pd.DataFrame
        dimension k x k
    """
    ix_slc = pd.IndexSlice

    # extract exposure matrix on one date
    pq_date = pd.to_datetime(date)
    pq_fname = FACTOR_COVARIANCE_FILES[pq_date.year]
    raw_cov_mat = pd.read_parquet(pq_fname, columns=[pq_date.__str__()])

    raw_cov_mat.index = raw_cov_mat.index.str.split("/", expand=True)
    raw_cov_mat.index.names = ["row_factorid", "col_factorid"]

    # subset to just the relevant factors
    if factorids is not None:
        fids = [fid.upper() for fid in factorids]
        raw_cov_mat = raw_cov_mat.loc[ix_slc[fids, fids], :]

    # pivot to matrix format
    cov_mat = (
        raw_cov_mat.reset_index()
        .pivot(index="row_factorid", columns="col_factorid", values=raw_cov_mat.columns.tolist())
        .droplevel(axis=1, level=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    cov_mat.index.name = "factorid"
    cov_mat.columns.name = "factorid"

    # convert from upper-triangular to symmetric
    cov_mat.replace(np.nan, 0, inplace=True)
    utm = cov_mat.to_numpy()
    cov_mat.loc[:, :] = np.where(utm, utm, utm.T)

    # Remove omitted factors
    if omitted_factors is not None:
        omitted_factors_upper = [x.upper() for x in omitted_factors]
        cov_mat = cov_mat.drop(omitted_factors_upper, axis=0)
        cov_mat = cov_mat.drop(omitted_factors_upper, axis=1)

    return cov_mat


def load_idio_risk_vector(
    date: str | pd.Timestamp, barrids: list[str] | pd.Index = None
) -> pd.DataFrame:
    """Load the specific risk vector from the consolidated barra parquet files.

    Parameters
    ----------
    date: str | pd.Timestamp
        '2010-03-04' or '20100304' format  or a pd.Timestamp
    barrids: list[str]
        list of barra ids

    Returns
    -------
    pd.DataFrame
        dimension N x 1
    """

    # extract exposure matrix on one date
    pq_date = pd.to_datetime(date)
    pq_fname = IDIOSYNCRATIC_VOL_FILES[pq_date.year]
    raw_spec_risk = pd.read_parquet(pq_fname, columns=[pq_date.__str__()])

    # subset to just the relevant barra ids
    if barrids is not None:
        raw_spec_risk = raw_spec_risk.loc[barrids, :]

    return raw_spec_risk.sort_index(axis=0)


# ------------------------------------------------------------------------------------------------ #
#                                    Auxiliary Functions
# ------------------------------------------------------------------------------------------------ #


def load_list_of_valid_barra_dates() -> list[str]:
    """Reads the dates in the Barra specific risk file and returns them in a list of strings.

    Returns
    -------
        list[str] of dates in yyyy-mm-dd format.
    """
    years = list(IDIOSYNCRATIC_VOL_FILES.keys())
    valid_dates = []
    for year in years:
        spec_risk_file = pq.ParquetFile(IDIOSYNCRATIC_VOL_FILES[year])
        schema = pa.schema([f.remove_metadata() for f in spec_risk_file.schema_arrow])
        dates = schema.names
        valid_dates.extend(
            [dt.replace(" 00:00:00", "") for dt in [dts for dts in dates if dts != "Barrid"]]
        )
    return [pd.Timestamp(date) for date in valid_dates]


def load_default_monthly_universe() -> pd.DataFrame:
    df = pd.read_parquet(RUSSELL_HISTORY_PARQUET)
    df = df.groupby(["date", "barrid"])["r3000_wt"].sum().reset_index()
    df = df.pivot(index="date", columns="barrid", values="r3000_wt") > 0
    return df
