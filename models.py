from series import *
import customtkinter as ctk
import torch
import numpy as np
import pandas as pd
from darts.models import NBEATSModel
from darts.models import NHiTSModel

class ModelRunWindow(ctk.CTkToplevel):
    def __init__(self, params, configs, file=None, series=None, predictions=[]):
        super().__init__()
        self.grab_set()
        self.grid_propagate(True)
        self.file = file
        if file:
            self.series = GetSeries(file)
        else:
            self.series = series
        self.params = params
        self.configurations = configs
        self.predictions = predictions
        self.model = None
        self.epochs = None

        #Barra de Progresso do Treinamento
        self.progress = ctk.CTkProgressBar(self)
        self.progress.pack(pady=20)
        self.progress.set(0)


    def model_train(self):
        print(self.series.train,self.series.valid)
        for epoch in range(self.epochs):
            self.model.fit(series=self.series.train, val_series=self.series.valid)
            self.update_progress(epoch)
        if self.params['save_checkpoints'] == 'True':
            self.model = NBEATSModel.load_from_checkpoint(self.model.model_name)
        print(self.model)

    def update_progress(self,epoch):
        progress_value = (epoch + 1) / self.epochs
        self.progress.set(progress_value)
        self.update_idletasks()

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

    def next_model_run(self):
        # Fecha modelo atual e abre próximo
        if self.configurations:
            next_model_run = self.configurations[0]
            self.configurations.pop(0)
            if next_model_run["model"] == "N-BEATS":
                ModelRunNBEATSWindow(next_model_run["parameters"], self.configurations, self.series)
            if next_model_run["model"] == "N-HiTS":
                ModelRunNHiTSWindow(next_model_run["parameters"], self.configurations, self.series)
            self.after(100, self.destroy)
        # else:
        #     ResultsWindow()
        #     self.after(100, self.destroy)

class ModelRunLHCWindow(ModelRunWindow):
    def __init__(self,params, series_file, configs):
        super().__init__(params, series_file, configs)
        self.title("LHC - Executando Modelo")
        self.centralize_window()
        self.bring_fwd_window()

        self.model_creation()
        self.after(100, self.model_train)
        self.evaluate_model()
        self.next_model_run()

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.model = LHCModel(**self.params)

class ModelRunNBEATSWindow(ModelRunWindow):
    def __init__(self,params, series_file, configs):
        super().__init__(params, series_file, configs)
        self.title("N-BEATS - Executando Modelo")
        self.centralize_window()
        self.bring_fwd_window()

        self.model_creation()
        self.after(100, self.model_train)
        self.evaluate_model()
        self.next_model_run()

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.model = NBEATSModel(**self.params)

class ModelRunNHiTSWindow(ModelRunWindow):
    def __init__(self,params, series_file, configs):
        super().__init__(params, series_file, configs)
        self.title("N-HiTS - Executando Modelo")
        self.centralize_window()
        self.bring_fwd_window()

        self.model_creation()
        self.after(100, self.model_train)
        self.evaluate_model()
        self.next_model_run()

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.model = NHiTSModel(**self.params)
