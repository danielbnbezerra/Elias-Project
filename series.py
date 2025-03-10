import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.tsatools import detrend
from darts import TimeSeries
from darts.utils.statistics import check_seasonality, plot_acf

class TreatSeries:
    def __init__(self, timeseries):
        self.timeseries = timeseries
        self.detrended_timeseries, self.timeseries_trend = self.remove_trend()
        self.timeseries_ratio = self.timeseries_ratio()


    def remove_trend(self, order=1): #Vários métodos de retirada de tendência? A Definir
        #detrended_series = TimeSeries.from_values(detrend(x=self.timeseries, order=order))
        timeseries_steps= np.arange(len(self.timeseries))
        coeffs = np.polyfit(timeseries_steps, self.timeseries, deg=order)
        trend_values = np.polyval(coeffs, timeseries_steps)

        # Detrend by subtracting the trend
        trend = TimeSeries.from_series(pd.Series(trend_values, self.timeseries.time_index))
        detrended_series = self.timeseries - trend
        return detrended_series, trend

    def timeseries_ratio(self):
        ratio_series = self.timeseries.pd_series().div(self.pd_series().shift(1))[1:]
        ratio_series = TimeSeries.from_pandas(ratio_series)
        return ratio_series