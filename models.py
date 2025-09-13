from series import *
from plotresults import *
from lhcmodel import *
import customtkinter as ctk
import torch
import numpy as np
import pandas as pd
from darts.models import NBEATSModel
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import Callback

class LossTracker(Callback):
    def __init__(self):
        self.losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the callback metrics
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.losses.append(loss.item())

class ModelRunWindow(ctk.CTkToplevel):
    def __init__(self, params, configs, series=None, predictions=None, losses=None):
        super().__init__()
        self.grab_set()
        self.grid_propagate(True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)  # Espaço antes dos botões
        self.rowconfigure(1, weight=1)
        self.series = series
        self.params = params
        self.configurations = configs
        self.predictions = predictions if predictions is not None else {}
        self.losses = losses if losses is not None else {}
        self.model = None
        self.epochs = None

        #Barra de Progresso do Treinamento
        self.progress = ctk.CTkProgressBar(self)
        self.progress.grid(row=0, column=0, pady=20)
        self.progress.set(0)

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
                ModelRunNBEATSWindow(params=next_model_run["parameters"], configs=self.configurations, series=self.series, preds=self.predictions, losses=self.losses)
            if next_model_run["model"] == "N-HiTS":
                ModelRunNHiTSWindow(params=next_model_run["parameters"], configs=self.configurations, series=self.series, preds=self.predictions, losses=self.losses)
            self.destroy()
        else:
            PlotWindow(self.series, self.predictions, self.losses)
            self.after(100, self.destroy)

    def centralize_window(self):
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
    def __init__(self,params, configs, series=None, preds=None, losses=None):
        super().__init__(params, configs, series, preds, losses)
        self.title("LHC - Executando Modelo")
        self.centralize_window()
        self.bring_fwd_window()

        self.device = set_device()
        self.seed = self.params["random_state"]
        if self.device == 'cuda':
            # Se estiver usando CUDA:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True  # Para consistência
            torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = False #Para desempenho
            # torch.backends.cudnn.benchmark = True
        else:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        self.model_creation()
        self.after(100, self.model_train)

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.model = LHCModel(input_size=self.params["input_size"],
                              hidden_size=self.params["hidden_size"],
                              num_layers=self.params["num_layers"],
                              output_size=self.params["output_size"]).to(self.device)
                              # random_state=self.params["random_state"])

    #A DEFINIR

    # def model_train(self):
    #     print(self.series.train,self.series.valid)
    #     for epoch in range(self.epochs):
    #         self.model.fit(series=self.series.train, val_series=self.series.valid)
    #         self.update_progress(epoch)
    #     self.predict_model()
    #
    # def predict_model(self):
    #     prediction = self.model.predict(series=self.series.train, n=len(self.series.valid))
    #     self.predictions["NBEATS"] = prediction
    #     self.losses["NBEATS"] = self.loss_tracker.losses
    #     self.next_model_run()

class ModelRunNBEATSWindow(ModelRunWindow):
    def __init__(self, params, configs, series=None, preds=None, losses=None):
        super().__init__(params, configs, series, preds, losses)
        self.title("N-BEATS - Executando Modelo")
        self.centralize_window()
        self.bring_fwd_window()

        self.loss_tracker = LossTracker()
        self.model_creation()
        self.after(100, self.model_train)

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.params["pl_trainer_kwargs"] = {"callbacks": [self.loss_tracker]}
        self.model = NBEATSModel(**self.params)

    def model_train(self):
        for attr, value in self.series.__dict__.items():
            print(f"{attr}: {value}")
        for epoch in range(self.epochs):
            self.model.fit(series=self.series.train_target,
                           past_covariates=self.series.train_cov,
                           val_past_covariates=self.series.valid_cov,
                           val_series=self.series.valid_target)
            self.update_progress(epoch)
        if self.params['save_checkpoints'] == 'true':
            self.model = NBEATSModel.load_from_checkpoint(self.model.model_name)
        self.predict_model()

    def predict_model(self):
        prediction = self.model.predict(series=self.series.train_target, past_covariates=self.series.prate_covariates, n=len(self.series.valid_target))
        self.predictions["NBEATS"] = prediction
        self.losses["NBEATS"] = self.loss_tracker.losses
        self.next_model_run()

class ModelRunNHiTSWindow(ModelRunWindow):
    def __init__(self, params, configs, series=None, preds=None, losses=None):
        super().__init__(params, configs, series, preds, losses)
        self.title("N-HiTS - Executando Modelo")
        self.centralize_window()
        self.bring_fwd_window()

        self.loss_tracker = LossTracker()
        self.model_creation()
        self.after(100, self.model_train)

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.params["pl_trainer_kwargs"] = {"callbacks": [self.loss_tracker]}
        self.model = NHiTSModel(**self.params)

    def model_train(self):

        for epoch in range(self.epochs):
            self.model.fit(series=self.series.train_target,
                           past_covariates=self.series.train_cov,
                           val_past_covariates=self.series.valid_cov,
                           val_series=self.series.valid_target)
            self.update_progress(epoch)
        if self.params['save_checkpoints'] == 'true':
            self.model = NHiTSModel.load_from_checkpoint(self.model.model_name)
        self.predict_model()

    def predict_model(self):
        prediction = self.model.predict(series=self.series.train_target, past_covariates=self.series.prate_covariates, n=len(self.series.valid_target))
        self.predictions["NHiTS"] = prediction
        self.losses["NHiTS"] = self.loss_tracker.losses
        self.next_model_run()
