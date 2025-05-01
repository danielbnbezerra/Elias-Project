import pandas as pd
#import numpy as np
#from statsmodels.tsa.tsatools import detrend
from darts import TimeSeries
from darts.metrics import mape,rmse
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

class MetricModels:
    def __init__(self, series, predictions):
        self.mape = {}
        self.rmse = {}
        self.generate_metrics(series,predictions)

    def generate_metrics(self, valid, predictions):

        models = list(predictions.keys())
        for model in models:
            if model == "LHC":
                self.mape["LHC"] = mape(valid, predictions["LHC"][:len(valid)]) #Passar a pred do LHC pra TimeSeries
                self.rmse["LHC"] = rmse(valid, predictions["LHC"][:len(valid)])
            if model == "NBEATS":
                self.mape["NBEATS"] = mape(valid, predictions["NBEATS"][:len(valid)])
                self.rmse["NBEATS"] = rmse(valid, predictions["NBEATS"][:len(valid)])
            if model == "NHiTS":
                self.mape["NHiTS"] = mape(valid, predictions["NHiTS"][:len(valid)])
                self.rmse["NHiTS"] = rmse(valid, predictions["NHiTS"][:len(valid)])