import queue
import threading

from multiprocessing import Queue
from queue import Empty
from darts import concatenate
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
    def __init__(self, params, configs, series=None, predictions=None, simulations=None, losses=None, models=None, residuals=None):
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
        self.simulations = simulations if simulations is not None else {}

        self.device = None
        self.model = None
        self.epochs = None
        self.current_epoch = None

        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.pack(pady=5)
        # Label de progresso
        self.label_progress = ctk.CTkLabel(self.progress_frame, text="Processando...")
        self.label_progress.grid(row=0, column=0, pady=(10, 0))

        # Barra de progresso
        self.progress = ctk.CTkProgressBar(self.progress_frame, mode="indeterminate")
        self.progress.grid(row=1, column=0, pady=20)
        self.progress.start()

    def training_finished(self):
        """Para a barra de progresso e atualiza a mensagem."""
        if self.winfo_exists():
            self.progress.stop()
            self.progress.configure(mode='determinate')
            self.progress.set(1)
            self.label_progress.configure(text="Processamento Concluído!")
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
                                     simul=self.simulations,
                                     losses=self.losses,
                                     models=self.models,
                                     residuals=self.residuals)
            if next_model_run["model"] == "N-HiTS":
                ModelRunNHiTSWindow(params=next_model_run["parameters"],
                                    configs=self.configurations,
                                    series=self.series,
                                    preds=self.predictions,
                                    simul=self.simulations,
                                    losses=self.losses,
                                    models=self.models,
                                    residuals=self.residuals)
            self.after(100, self.destroy)
        else:
            PlotWindow(self.series, self.predictions, self.simulations, self.residuals, self.losses, self.models)
            self.after(100, self.destroy)

    def recursive_simulation(self, model, model_type, device=None):
        """
        Simula recursivamente, corrigindo a frequência da série 'just-in-time'.
        """
        input_len = 24

        original_flow = self.series.flow
        original_covariates = self.series.prate_covariates

        flow_corrected = TimeSeries.from_times_and_values(
            original_flow.time_index,
            original_flow.values(),
            freq='D',
            fill_missing_dates=True # Adicionado para robustez
        )
        covariates_corrected = TimeSeries.from_times_and_values(
            original_covariates.time_index,
            original_covariates.values(),
            freq='D',
            fill_missing_dates=True # Adicionado para robustez
        )

        if model_type == 'lhc':
            # A lógica do LHC também se beneficia da série com frequência correta
            start_flow_tensor = torch.tensor(flow_corrected[:input_len].values(), dtype=torch.float32).unsqueeze(0)
            prate_cov_tensor = torch.tensor(covariates_corrected.values(), dtype=torch.float32).unsqueeze(0)

            start_cov_tensor = prate_cov_tensor[:, :input_len, :]
            initial_input_seq_sim = torch.cat([start_flow_tensor, start_cov_tensor], dim=2)

            future_covariates_sim = prate_cov_tensor[:, input_len:, :]
            n_predict = len(flow_corrected) - input_len

            preds_tensor = recursive_predict(
                model,
                initial_input_seq_sim,
                future_covariates_sim,
                n_predict,
                device
            )
            simul_series = predictions_to_timeseries(preds_tensor, self.series.flow[input_len:].time_index)
            return simul_series


        elif model_type in ['nbeats', 'nhits']:

            simulated_values = []

            current_input_series = flow_corrected[:input_len]

            all_covariates = covariates_corrected

            n_predict = len(flow_corrected) - input_len

            for i in range(n_predict):
                covariates_for_prediction = all_covariates.slice_intersect(current_input_series)
                pred = model.predict(
                    n=1,
                    series=current_input_series,
                    past_covariates=covariates_for_prediction
                )
                simulated_values.append(pred)
                current_input_series = current_input_series.append(pred)[-input_len:]

            simulated_series = concatenate(simulated_values, axis=0)
            return simulated_series

        else:
            raise ValueError("model_type deve ser 'lhc', 'nbeats' ou 'nhits'.")

    def centralize_window(self):
        window_width = 400
        window_height = 200
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

class ModelRunLHCWindow(ModelRunWindow):
    def __init__(self, params, configs, series, preds=None, simul=None, losses=None, models=None, residuals=None):
        super().__init__(params, configs, series, preds, simul, losses, models, residuals)
        self.title("LHC - Executando Modelo")
        self.centralize_window()

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

        callbacks = [self.loss_tracker]

        if self.params.get('save_checkpoints', 'false') == 'true':
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
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
        # Prepara a janela de aquecimento (warm-up) com os últimos dados de treino
        train_len = self.train_target.shape[1]
        warm_up_flow = self.train_target[:, train_len - self.input_len:, :]
        warm_up_cov = self.train_cov[:, train_len - self.input_len:, :]
        initial_input_seq_pred = torch.cat([warm_up_flow, warm_up_cov], dim=2)

        # As covariáveis futuras para a predição são as do conjunto de validação
        future_covariates_pred = self.valid_cov

        preds_tensor = recursive_predict(
            self.model,
            initial_input_seq_pred,
            future_covariates_pred,
            n_predict,
            self.device
        )

        preds_ts = predictions_to_timeseries(preds_tensor, self.pred_time_index)
        self.label_progress.configure(text="Processando os resíduos. Aguarde.")
        residuals_ts = residuals_timeseries(self.valid_target, preds_ts)
        self.predictions["LHC"] = preds_ts
        self.residuals["LHC"] = residuals_ts

        simulation_ts = self.recursive_simulation(self.model, 'lhc', self.device)
        self.simulations["LHC"] = simulation_ts

        self.losses["LHC"] = self.loss_tracker.losses
        self.models["LHC"] = self.model
        self.training_finished()

        self.after(1000, self.next_model_run)

class ModelRunDartsWindow(ModelRunWindow):
    def __init__(self, params, configs, series, model_class, model_name, preds=None, simul=None, losses=None,
                 models=None, residuals=None):
        super().__init__(params, configs, series, preds, simul, losses, models, residuals)
        self.title(f"{model_name} - Executando Modelo")
        self.centralize_window()

        self.model_class = model_class  # Ex: NBEATSModel ou NHiTSModel
        self.model_name = model_name  # Ex: "N-BEATS" ou "N-HiTS"
        self.epochs = self.params["n_epochs"]

        # Prepara o modelo com os callbacks
        self.loss_tracker = LossTracker()

        self.params["pl_trainer_kwargs"] = {
            "callbacks": [self.loss_tracker],
            "enable_progress_bar": False
        }

        self.model = self.model_class(**self.params)

        # Inicia o treinamento em uma thread para não travar a GUI
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        """
        Executa o treinamento completo do modelo em uma única chamada fit().
        """
        # A Darts/Pytorch Lightning cuidará de iterar por todas as épocas
        self.model.fit(series=self.series.train_target,
                       past_covariates=self.series.train_cov,
                       val_series=self.series.valid_target,
                       val_past_covariates=self.series.valid_cov)

        # Após o fim do treinamento, agenda a execução do pós-treino na thread principal da GUI
        self.after(0, self.post_training)

    def post_training(self):

        if self.params.get('save_checkpoints', 'false') == 'true':
            self.model = self.model_class.load_from_checkpoint(self.model.model_name)

        # Predição
        predictions = self.model.predict(series=self.series.train_target,
                                         past_covariates=self.series.prate_covariates,
                                         n=len(self.series.valid_target))
        self.predictions[self.model_name] = predictions

        # Simulação
        simulation = self.recursive_simulation(self.model,
                                               self.model_name.lower().replace('-', ''))  # 'n-beats' -> 'nbeats'
        self.simulations[self.model_name] = simulation

        # Resíduos
        self.update_idletasks()
        residuals = self.model.residuals(self.series.valid_target,
                                         past_covariates=self.series.valid_cov,
                                         verbose=False,
                                         retrain=False)
        self.residuals[self.model_name] = residuals

        # Salva resultados e finaliza
        self.losses[self.model_name] = self.loss_tracker.losses
        self.models[self.model_name] = self.model
        self.training_finished()

        self.after(1000, self.next_model_run)

class ModelRunNBEATSWindow(ModelRunDartsWindow):
    def __init__(self, params, configs, series, preds=None, simul=None, losses=None, models=None, residuals=None):
        super().__init__(params=params,
                         configs=configs,
                         series=series,
                         model_class=NBEATSModel,
                         model_name="N-BEATS",
                         preds=preds,
                         simul=simul,
                         losses=losses,
                         models=models,
                         residuals=residuals)

class ModelRunNHiTSWindow(ModelRunDartsWindow):
    def __init__(self, params, configs, series, preds=None, simul=None, losses=None, models=None, residuals=None):
        super().__init__(params=params,
                         configs=configs,
                         series=series,
                         model_class=NHiTSModel,
                         model_name="N-HiTS",
                         preds=preds,
                         simul=simul,
                         losses=losses,
                         models=models,
                         residuals=residuals)