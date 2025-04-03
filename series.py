import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#from statsmodels.tsa.tsatools import detrend
from darts import TimeSeries
#from darts.utils.statistics import check_seasonality, plot_acf

class GetSeries:
    def __init__(self, series_file):
        self.timeseries = self.make_timeseries(series_file)
        self.train, self.valid = self.train_valid_split()

    def make_timeseries(self, series_file):
        df = pd.read_csv(series_file)
        timeseries = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
        return timeseries

    def train_valid_split(self):
        train, valid = self.timeseries.split_before(int(self.timeseries.n_timesteps * 0.80)) #Por padrão separando em 80% treino e 20% validação
        return train, valid


