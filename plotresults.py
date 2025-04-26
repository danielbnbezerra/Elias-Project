import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from darts import TimeSeries

from series import MetricModels

#FALTA INTEGRAR EXEMPLO 20 AQUI

class PlotWindow(ctk.CTkToplevel):
    def __init__(self, series, preds, losses, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.grab_set()
        self.grid_propagate(True)
        self.timeseries = series.timeseries
        self.predictions = preds
        self.loss_curves = losses
        self.metrics = MetricModels(series.valid, preds)

    def centralize_window(self):
        # window_width = round(self.winfo_width(),-1)
        # window_height = round(self.winfo_height(),-1)
        window_width = 400
        window_height = 200
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

    def bring_fwd_window(self):
        self.attributes("-topmost", True)