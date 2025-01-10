import os
from abc import ABC, abstractmethod

import pandas as pd

DATA_DIR = ""


class Dataset(ABC):
    def __init__(self, RAW_FILE_PATH: str, CLEAN_FILE_PATH: str) -> None:
        if not DATA_DIR:
            raise "No data directory in root!"

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        if not os.path.exists(CLEAN_FILE_PATH):
            if not os.path.exists(RAW_FILE_PATH):
                print("DOWNLOADING RAW FILE")
                self.download()

            print("CLEANING RAW FILE")
            self.clean()

        print("LOADING CLEAN FILE")
        self.df = pd.read_parquet(CLEAN_FILE_PATH)
        self.attributes()

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def clean(self):
        pass

    def attributes(self):
        pass
