from series import *
import customtkinter as ctk
import torch
import numpy as np
import pandas as pd
from darts.models import NBEATSModel
from darts.models import NHiTSModel

class ModelRunWindow(ctk.CTkToplevel):
    def __init__(self, params, file):
        super().__init__()
        self.grab_set()
        self.grid_propagate(True)
        self.series = GetSeries(file)
        self.params = params
        self.model = None
        self.centralize_window()

    def model_train(self):
        return None
        # self.model.fit(self.data_train, val_series=self.data_valid)
        # eval_model(model, len(val_scaled) + 48, series_scaled, val_scaled)
        # print('best_model')
        # if self.params['save_checkpoints'] == 'True':
        #     self.model = NBEATSModel.load_from_checkpoint(self.model.model_name)
        # eval_model(model, len(val_scaled) + 48, series_scaled, val_scaled)

    def evaluate_model(self):

        return None

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


# class ModelRunLHCWindow(ModelRunWindow):
#     def __init__(self,params):
#         super().__init__(params)
#         self.title("LHC - Executando Modelo")
#
#     def model_creation(self):
#         self.model = LHCModel(**self.params)

class ModelRunNBEATSWindow(ModelRunWindow):
    def __init__(self,params, series_file):
        super().__init__(params, series_file)
        self.title("N-BEATS - Executando Modelo")

    def model_creation(self):
        self.model = NBEATSModel(**self.params)

class ModelRunNHiTSWindow(ModelRunWindow):
    def __init__(self,params,file):
        super().__init__(params,file)
        self.title("N-HiTS - Executando Modelo")

    def model_creation(self):
        self.model = NHiTSModel(**self.params)