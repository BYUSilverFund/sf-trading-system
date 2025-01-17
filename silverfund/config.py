# Description: Configuration file for silverfund
# Author: Seth Peterson
# Last Updated: October 2023

import os

from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# GENERAL CONFIGURATION
# ----------------------------

RESTRICT_DOMESTIC = True  # bool, True = restrict to domestic securities, False = do not restrict
RESTRICT_BARRA_HISTORY = True  # bool, True = restrict to BARRA history, False = do not restrict
CONFIG_VALIDATE_FILES = False  # bool, True = validate files, False = do not validate files
VERBOSE = True  # bool, True = print all messages, False = print only errors
VERBOSE_LEVEL = 2  # 0 = Minimal, 1 = File I/O, 2 = Full

# ----------------------------
# VERBOSE SETUP
# ----------------------------

VERBOSE_LEVEL = 0 if not VERBOSE else VERBOSE_LEVEL

# ----------------------------
# FILE LOCATIONS CONFIGURATION
# ----------------------------

# Directory to all datafiles
USER = os.getenv("ROOT").split("/")[2]
ROOT_DIR = f"/home/{USER}"
FULTON_DATA_DIR = ROOT_DIR + "/groups/"

# Compiled Data Directory
GRP_QUANT_DIR = FULTON_DATA_DIR + "grp_quant/"

# Data Directory
DATA_DIR = GRP_QUANT_DIR + "data/"

# BARRA CUSIP PERMNO MAP
IDENTIFIER_MAPPING_CSV = DATA_DIR + "usslow_ids.csv"

# COVARIANCE MATRICES FILES
USSLOW_DIR = DATA_DIR + "barra_usslow/"
IDIOSYNCRATIC_VOL_PREFIX = "spec_risk_"
FACTOR_EXPOSURES_PREFIX = "exposures_"
FACTOR_COVARIANCE_PREFIX = "factor_covariance_"

# RETURN FILES
USSLOW_RETS_DIR = DATA_DIR + "barra_usslow_ret/"
BARRA_RET_PREFIX = "ret_"

# Monthly default universe file
RUSSELL_HISTORY_PARQUET = DATA_DIR + "russell_history.parquet"
MSF_RUSSELL_3000_PARQUET = DATA_DIR + "mega_monthly.parquet"

# ----------------------------
# Configuration Setup
# ----------------------------

# Maps year to the corresponding file

# Covariance Maps
IDIOSYNCRATIC_VOL_FILES = {}
FACTOR_EXPOSURE_FILES = {}
FACTOR_COVARIANCE_FILES = {}

# Return Maps
BARRA_RET_FILES = {}

# Populate the maps
for filename in os.listdir(USSLOW_DIR.replace("~", os.path.expanduser("~"))):
    if filename.startswith(IDIOSYNCRATIC_VOL_PREFIX):
        try:
            year = int(filename[len(IDIOSYNCRATIC_VOL_PREFIX) : len(IDIOSYNCRATIC_VOL_PREFIX) + 4])
            IDIOSYNCRATIC_VOL_FILES[year] = USSLOW_DIR + filename
        except:
            print(f"Error parsing year from {filename}")

    elif filename.startswith(FACTOR_EXPOSURES_PREFIX):
        try:
            year = int(filename[len(FACTOR_EXPOSURES_PREFIX) : len(FACTOR_EXPOSURES_PREFIX) + 4])
            FACTOR_EXPOSURE_FILES[year] = USSLOW_DIR + filename
        except:
            print(f"Error parsing year from {filename}")

    elif filename.startswith(FACTOR_COVARIANCE_PREFIX):
        try:
            year = int(filename[len(FACTOR_COVARIANCE_PREFIX) : len(FACTOR_COVARIANCE_PREFIX) + 4])
            FACTOR_COVARIANCE_FILES[year] = USSLOW_DIR + filename
        except:
            print(f"Error parsing year from {filename}")

for filename in os.listdir(USSLOW_RETS_DIR.replace("~", os.path.expanduser("~"))):
    if filename.startswith(BARRA_RET_PREFIX):
        try:
            year = int(filename[len(BARRA_RET_PREFIX) : len(BARRA_RET_PREFIX) + 4])
            BARRA_RET_FILES[year] = USSLOW_RETS_DIR + filename
        except:
            print(f"Error parsing year from {filename}")
