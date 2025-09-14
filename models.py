import threading

from darts.models import NBEATSModel
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader

from lhcmodel import *
from plotresults import *


class LossTracker(Callback):
    def __init__(self):
        self.losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the callback metrics
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.losses.append(loss.item())

class ModelRunWindow(ctk.CTkToplevel):
    def __init__(self, params, configs, series=None, predictions=None, losses=None, models=None, residuals=None):
        super().__init__()
        self.grab_set()
        self.grid_propagate(True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)  # Espaço antes dos botões
        self.rowconfigure(1, weight=1)
        self.series = series
        self.configurations = configs
        self.params = params
        self.predictions = predictions if predictions is not None else {}
        self.losses = losses if losses is not None else {}
        self.models = models if models is not None else {}
        self.residuals = residuals if residuals is not None else {}

        self.model = None
        self.epochs = None
        self.current_epoch = None

        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.pack(pady=5)
        # Label de progresso
        self.label_progress = ctk.CTkLabel(self.progress_frame, text="Processando. 0% concluído.")
        self.label_progress.grid(row=0, column=0, pady=(10, 0))

        # Barra de progresso
        self.progress = ctk.CTkProgressBar(self.progress_frame)
        self.progress.grid(row=1, column=0, pady=20)
        self.progress.set(0)

    def update_progress(self,epoch):
        progress_value = (epoch + 1) / self.epochs
        percent_progress = int(progress_value * 100)
        self.progress.set(progress_value)
        self.label_progress.configure(text=f"Processando. {percent_progress}% concluído")
        self.update_idletasks()

    def next_model_run(self):
        # Fecha modelo atual e abre próximo
        if self.configurations:
            next_model_run = self.configurations[0]
            self.configurations.pop(0)
            if next_model_run["model"] == "N-BEATS":
                ModelRunNBEATSWindow(params=next_model_run["parameters"],
                                     configs=self.configurations,
                                     series=self.series,
                                     preds=self.predictions,
                                     losses=self.losses,
                                     models=self.models,
                                     residuals=self.residuals)
            if next_model_run["model"] == "N-HiTS":
                ModelRunNHiTSWindow(params=next_model_run["parameters"],
                                    configs=self.configurations,
                                    series=self.series,
                                    preds=self.predictions,
                                    losses=self.losses,
                                    models=self.models,
                                    residuals=self.residuals)
            self.destroy()
        else:
            for nome_modelo, data in self.predictions.items():
                print(f"{nome_modelo}: {data}")
            for nome_modelo, data in self.residuals.items():
                print(f"{nome_modelo}: {data}")
            for nome_modelo, data in self.losses.items():
                print(f"{nome_modelo}: {data}")
            for nome_modelo, data in self.models.items():
                print(f"{nome_modelo}: {data}")

            PlotWindow(self.series, self.predictions, self.residuals, self.losses, self.models)
            self.after(100, self.destroy)

    def centralize_window(self):
        window_width = 400
        window_height = 200
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

    # def bring_fwd_window(self):
    #     self.attributes("-topmost", True)


class ModelRunLHCWindow(ModelRunWindow):
    def __init__(self, params, configs, series, preds=None, losses=None, models=None, residuals=None):
        super().__init__(params, configs, series, preds, losses, models, residuals)
        self.title("Treinamento LHCModel")
        self.centralize_window()
        # self.bring_fwd_window()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tensor sets baseados em GetSeries (prep treino e validação)
        self.train_target = torch.tensor(self.series.train_target.values(), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.valid_target = torch.tensor(self.series.valid_target.values(), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.train_cov = torch.tensor(self.series.train_cov.values(), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.valid_cov = torch.tensor(self.series.valid_cov.values(), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.prate_covariates = torch.tensor(self.series.prate_covariates.values(), dtype=torch.float32).unsqueeze(0).to(self.device)

        self.pred_time_index = self.series.valid_target.time_index

        self.input_len = params['input_length']  # Para janelas de entrada
        self.output_len = params['output_length']  # Quantos passos para treino (normalmente 1)

        train_input = torch.cat([self.train_target, self.train_cov], dim=2)
        self.inputs, self.targets = series_to_batches([train_input.cpu()], self.input_len, self.output_len)
        self.inputs = self.inputs.to(self.device)
        self.targets = self.targets.to(self.device)

        self.epochs = self.params['n_epochs']

        # Inicializa o modelo Lightning e callback
        self.model = LHCModel(self.params, self.inputs.shape[2])
        self.loss_tracker = LossTracker()

        # Thread para treino não travar GUI
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        train_dataset = TensorDataset(self.inputs, self.targets)
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)

        valid_input = torch.cat([self.valid_target, self.valid_cov], dim=2)
        valid_inputs, valid_targets = series_to_batches([valid_input.cpu()], self.input_len, self.output_len)
        valid_inputs = valid_inputs.to(self.device)
        valid_targets = valid_targets.to(self.device)
        valid_dataset = TensorDataset(valid_inputs, valid_targets)
        valid_loader = DataLoader(valid_dataset, batch_size=self.params['batch_size'])

        callbacks = [self.loss_tracker, ProgressBarCallback(self)]

        if self.params.get('save_checkpoints', 'false').lower() == 'true':
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",  # Agora monitoramos a loss de validação
                mode="min",
                save_top_k=1,
                filename="lhc-best-model-{epoch:02d}-{val_loss:.4f}",
                verbose=True
            )
            callbacks.append(checkpoint_callback)
        else:
            checkpoint_callback = None

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(self.model, train_loader, valid_loader)

        if checkpoint_callback and checkpoint_callback.best_model_path:
            self.model = LHCModel.load_from_checkpoint(checkpoint_callback.best_model_path,input_size=self.inputs.shape[2])
            self.model.to(self.device)

        self.after(0, self.post_training)

    def post_training(self):
        n_predict = len(self.valid_target.squeeze(0))

        preds_tensor = recursive_predict(
            self.model,
            self.train_target,
            self.prate_covariates,
            self.input_len,
            n_predict,
            self.device
        )

        preds_ts = predictions_to_timeseries(preds_tensor, self.pred_time_index)
        residuals_ts = residuals_timeseries(self.valid_target, preds_ts)

        self.predictions["LHC"] = preds_ts
        self.residuals["LHC"] = residuals_ts
        self.losses["LHC"] = self.loss_tracker.losses
        self.models["LHC"] = self.model
        self.next_model_run()

class ModelRunNBEATSWindow(ModelRunWindow):
    def __init__(self, params, configs, series=None, preds=None, losses=None, models=None, residuals=None):
        super().__init__(params, configs, series, preds, losses, models, residuals)
        self.title("N-BEATS - Executando Modelo")
        self.centralize_window()
        # self.bring_fwd_window()

        self.loss_tracker = LossTracker()
        self.model_creation()
        self.after(100, self.start_model_train)

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.params["pl_trainer_kwargs"] = {"callbacks": [self.loss_tracker]}
        self.model = NBEATSModel(**self.params)

    def start_model_train(self):
        """Inicializa o treino assíncrono"""
        self.current_epoch = 0
        self.progress.set(0)
        self.label_progress.configure(text="Processando. 0% concluído")
        self.run_model_epoch()

    def run_model_epoch(self):
        """Treina uma época sem travar a GUI"""
        if self.current_epoch < self.epochs:
            # Treina a época atual
            self.model.fit(series=self.series.train_target,
                           past_covariates=self.series.train_cov,
                           val_past_covariates=self.series.valid_cov,
                           val_series=self.series.valid_target)

            # Atualiza barra e label
            self.update_progress(self.current_epoch)
            #Próxima época
            self.current_epoch += 1
            #Agenda a próxima época
            self.after(1, self.run_model_epoch)
        else:
            self.label_progress.configure(text="Processando. 100% concluído")
            self.after_training_done()

    def after_training_done(self):
        """Ações após treino finalizado"""
        if self.params.get('save_checkpoints','false') == 'true':
            self.model = NBEATSModel.load_from_checkpoint(self.model.model_name)
        self.predict_model()

    def compute_residuals(self):
        self.residuals["NBEATS"] = self.model.residuals(self.series.valid_target, past_covariates=self.series.valid_cov, verbose=True, retrain=False)
        self.after(0, self.residuals_done)

    def residuals_done(self):
        self.label_progress.configure(text="Concluído!")
        self.losses["NBEATS"] = self.loss_tracker.losses
        self.models["NBEATS"] = self.model
        self.after(1000, self.next_model_run)

    def start_residuals(self):
        threading.Thread(target=self.compute_residuals, daemon=True).start()

    def predict_model(self):
        predictions = self.model.predict(series=self.series.train_target, past_covariates=self.series.prate_covariates, n=len(self.series.valid_target))
        self.predictions["NBEATS"] = predictions
        self.label_progress.configure(text="Processando os resíduos. Aguarde.")
        self.start_residuals()

class ModelRunNHiTSWindow(ModelRunWindow):
    def __init__(self, params, configs, series=None, preds=None, losses=None, models=None, residuals=None):
        super().__init__(params, configs, series, preds, losses, models, residuals)
        self.title("N-HiTS - Executando Modelo")
        self.centralize_window()
        # self.bring_fwd_window()

        self.loss_tracker = LossTracker()
        self.model_creation()
        self.after(100, self.start_model_train)

    def model_creation(self):
        self.epochs = self.params["n_epochs"]
        self.params["n_epochs"] = 1
        self.params["pl_trainer_kwargs"] = {"callbacks": [self.loss_tracker]}
        self.model = NHiTSModel(**self.params)

    def start_model_train(self):
        self.current_epoch = 0
        self.progress.set(0)
        self.label_progress.configure(text="Processando. 0% concluído")
        self.run_model_epoch()

    def run_model_epoch(self):
        if self.current_epoch < self.epochs:
            # Treina a época atual
            self.model.fit(series=self.series.train_target,
                           past_covariates=self.series.train_cov,
                           val_past_covariates=self.series.valid_cov,
                           val_series=self.series.valid_target)

            # Atualiza barra e label
            self.update_progress(self.current_epoch)
            # Próxima época
            self.current_epoch += 1
            # Agenda a próxima época
            self.after(1, self.run_model_epoch)
        else:
            self.label_progress.configure(text="Processando. 100% concluído")
            self.after_training_done()

    def after_training_done(self):
        """Ações após treino finalizado"""
        if self.params.get('save_checkpoints','false') == 'true':
            self.model = NHiTSModel.load_from_checkpoint(self.model.model_name)
        self.predict_model()

    def compute_residuals(self):
        self.residuals["NHiTS"] = self.model.residuals(self.series.valid_target, past_covariates=self.series.valid_cov,
                                                        verbose=True, retrain=False)
        self.after(0, self.residuals_done)

    def residuals_done(self):
        self.label_progress.configure(text="Concluído!")
        self.losses["NHiTS"] = self.loss_tracker.losses
        self.models["NHiTS"] = self.model
        self.after(1000, self.next_model_run)

    def start_residuals(self):
        threading.Thread(target=self.compute_residuals, daemon=True).start()

    def predict_model(self):
        predictions = self.model.predict(series=self.series.train_target, past_covariates=self.series.prate_covariates,
                                         n=len(self.series.valid_target))
        self.predictions["NHiTS"] = predictions
        self.label_progress.configure(text="Processando os resíduos. Aguarde.")
        self.start_residuals()