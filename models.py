from series import *
import customtkinter as ctk
import torch
import numpy as np
import pandas as pd
from darts.models import NBEATSModel
from darts.models import NHiTSModel

class ModelRunWindow(ctk.CTkToplevel):
    def __init__(self, params, configs, series_file=None, series=None, predictions=None):
        super().__init__()
        self.grab_set()
        self.grid_propagate(True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)  # Espaço antes dos botões
        self.rowconfigure(1, weight=1)
        self.file = series_file
        if series_file:
            self.series = GetSeries(series_file)
        else:
            self.series = series
        self.params = params
        self.configurations = configs
        if predictions:
            self.predictions = predictions
        else:
            self.predictions=[]
        self.model = None
        self.epochs = None

        #Barra de Progresso do Treinamento
        self.progress = ctk.CTkProgressBar(self)
        self.progress.grid(row=0, column=0, pady=20)
        self.progress.set(0)


    def model_train(self):
        print(self.series.train,self.series.valid)
        for epoch in range(self.epochs):
            self.model.fit(series=self.series.train, val_series=self.series.valid)
            self.update_progress(epoch)
        if self.params['save_checkpoints'] == 'true':
            self.model = NBEATSModel.load_from_checkpoint(self.model.model_name)
        self.predict_model()

    def update_progress(self,epoch):
        progress_value = (epoch + 1) / self.epochs
        self.progress.set(progress_value)
        self.update_idletasks()

    def next_model_run(self):
        # Fecha modelo atual e abre próximo
        if self.configurations:
            next_model_run = self.configurations[0]
            self.configurations.pop(0)
            if next_model_run["model"] == "N-BEATS":
                ModelRunNBEATSWindow(params=next_model_run["parameters"], configs=self.configurations, series=self.series, preds=self.predictions)
            if next_model_run["model"] == "N-HiTS":
                ModelRunNHiTSWindow(params=next_model_run["parameters"], configs=self.configurations, series=self.series, preds=self.predictions)
            self.destroy()
        # else:
        #     ResultsWindow()
        #     self.after(100, self.destroy)

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

class ModelRunLHCWindow(ModelRunWindow):
    def __init__(self,params, configs, series_file=None, series=None, preds=None):
        super().__init__(params, configs, series_file, series, preds)
        self.title("LHC - Executando Modelo")
    #     self.centralize_window()
    #     self.bring_fwd_window()
    #
    #     self.model_creation()
    #     self.after(100, self.model_train)
    #     self.evaluate_model()
    #     self.next_model_run()
    #
    # def model_creation(self):
    #     self.epochs = self.params["n_epochs"]
    #     self.params["n_epochs"] = 1
    #     self.model = LHCModel(**self.params)

class ModelRunNBEATSWindow(ModelRunWindow):
    def __init__(self, params, configs, series_file=None, series=None, preds=None):
        super().__init__(params, configs, series_file, series, preds)
        self.title("N-BEATS - Executando Modelo")
        self.centralize_window()
        self.bring_fwd_window()

        self.model_creation()
        self.after(100, self.model_train)

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.model = NBEATSModel(**self.params)

    def predict_model(self):
        prediction= self.model.predict(series=self.series.train, n=len(self.series.valid))
        print(prediction)
        self.predictions.append({"NBEATS":prediction})
        print(self.predictions)
        self.next_model_run()

class ModelRunNHiTSWindow(ModelRunWindow):
    def __init__(self, params, configs, series_file=None, series=None, preds=None):
        super().__init__(params, configs, series_file, series, preds)
        self.title("N-HiTS - Executando Modelo")
        self.centralize_window()
        self.bring_fwd_window()

        self.model_creation()
        self.after(100, self.model_train)

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.model = NHiTSModel(**self.params)

    def predict_model(self):
        prediction= self.model.predict(series=self.series.train, n=len(self.series.valid))
        print(prediction)
        self.predictions.append({"NHiTS":prediction})
        print(self.predictions)
        self.next_model_run()
