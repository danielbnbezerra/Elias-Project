import threading

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
    def __init__(self, params, configs, series=None, predictions=None, simulations=None, losses=None, models=None, residuals=None, all_params=None):
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
        self.all_params = all_params if all_params is not None else {}

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
                                     residuals=self.residuals,
                                     all_params=self.all_params)
            if next_model_run["model"] == "N-HiTS":
                ModelRunNHiTSWindow(params=next_model_run["parameters"],
                                    configs=self.configurations,
                                    series=self.series,
                                    preds=self.predictions,
                                    simul=self.simulations,
                                    losses=self.losses,
                                    models=self.models,
                                    residuals=self.residuals,
                                    all_params=self.all_params)
            self.after(100, self.destroy)
        else:
            PlotWindow(self.series, self.predictions, self.simulations, self.residuals, self.losses, self.models, self.all_params)
            self.after(100, self.destroy)

    # def recursive_simulation(self, model, model_type, input_len, device=None):
    #     """
    #     Simula recursivamente, corrigindo a frequência da série 'just-in-time'.
    #     """
    #
    #     scaled_flow = self.series.scaled_flow
    #     scaled_covariates = self.series.scaled_prate_covariates
    #
    #     flow_corrected = TimeSeries.from_times_and_values(
    #         scaled_flow.time_index,
    #         scaled_flow.values(),
    #         freq='D',
    #         fill_missing_dates=True # Adicionado para robustez
    #     )
    #     covariates_corrected = TimeSeries.from_times_and_values(
    #         scaled_covariates.time_index,
    #         scaled_covariates.values(),
    #         freq='D',
    #         fill_missing_dates=True # Adicionado para robustez
    #     )
    #
    #     if model_type == 'lhc':
    #         # A lógica do LHC também se beneficia da série com frequência correta
    #         start_flow_tensor = torch.tensor(flow_corrected[:input_len].values(), dtype=torch.float32).unsqueeze(0)
    #         prate_cov_tensor = torch.tensor(covariates_corrected.values(), dtype=torch.float32).unsqueeze(0)
    #
    #         start_cov_tensor = prate_cov_tensor[:, :input_len, :]
    #         initial_input_seq_sim = torch.cat([start_flow_tensor, start_cov_tensor], dim=2)
    #
    #         future_covariates_sim = prate_cov_tensor[:, input_len:, :]
    #         n_predict = len(flow_corrected) - input_len
    #
    #         preds_tensor = recursive_predict(
    #             model,
    #             initial_input_seq_sim,
    #             future_covariates_sim,
    #             n_predict,
    #             device
    #         )
    #
    #         simul_series = predictions_to_timeseries(preds_tensor, self.series.flow[input_len:].time_index)
    #         return simul_series
    #
    #     elif model_type in ['nbeats', 'nhits']:
    #
    #         simulated_values = []
    #
    #         current_input_series = flow_corrected[:input_len]
    #
    #         all_covariates = covariates_corrected
    #
    #         n_predict = len(flow_corrected) - input_len
    #
    #         for i in range(n_predict):
    #             covariates_for_prediction = all_covariates.slice_intersect(current_input_series)
    #             pred = model.predict(
    #                 n=1,
    #                 series=current_input_series,
    #                 past_covariates=covariates_for_prediction
    #             )
    #             simulated_values.append(pred)
    #             current_input_series = current_input_series.append(pred)[-input_len:]
    #
    #         simulated_series = concatenate(simulated_values, axis=0)
    #         return simulated_series
    #
    #     else:
    #         raise ValueError("model_type deve ser 'lhc', 'nbeats' ou 'nhits'.")

    def backtest_simulation(self, model, model_type):
        """
        Simula recursivamente, corrigindo a frequência da série 'just-in-time'.
        """

        scaled_flow = self.series.scaled_flow
        scaled_covariates = self.series.scaled_prate_covariates

        if model_type == 'lhc':
            """
                    Executa um backtesting manual para o modelo LHC, simulando historical_forecasts.
                    """
            # Parâmetros do modelo e da série
            input_len = model.hparams.input_length
            device = model.device
            forecast_horizon = 1
            stride = 1

            # Usaremos as séries completas normalizadas para o processo de slicing


            # Ponto de início: o índice logo após o fim do período de treino
            start_index = input_len

            all_forecasts = []

            # Loop que "desliza" pelo período de validação
            for i in range(start_index, len(scaled_flow) - forecast_horizon + 1, stride):
                # --- Preparação dos Dados para cada Previsão ---

                # 1. Pega a janela de "aquecimento" (warm-up) que antecede o ponto da previsão
                history_start_index = i - input_len
                history_end_index = i

                warm_up_flow_tensor = torch.tensor(scaled_flow[history_start_index:history_end_index].values(),
                                                   dtype=torch.float32).unsqueeze(0)
                warm_up_cov_tensor = torch.tensor(
                    scaled_covariates[history_start_index:history_end_index].values(),
                    dtype=torch.float32).unsqueeze(0)
                initial_input_seq = torch.cat([warm_up_flow_tensor, warm_up_cov_tensor], dim=2)

                # 2. Pega as covariáveis futuras conhecidas para o horizonte da previsão
                future_cov_start_index = i
                future_cov_end_index = i + forecast_horizon
                future_covariates_tensor = torch.tensor(
                    scaled_covariates[future_cov_start_index:future_cov_end_index].values(),
                    dtype=torch.float32).unsqueeze(0)

                # --- Chamada da Função de Previsão ---

                # Chama sua função original para prever apenas o horizonte desejado (ex: 1 passo)
                pred_tensor = recursive_predict(
                    model,
                    initial_input_seq.to(device),
                    future_covariates_tensor.to(device),
                    n_predict=forecast_horizon,
                    device=device
                )

                # --- Armazenamento do Resultado ---

                # Converte o tensor previsto para um objeto TimeSeries com a data correta
                pred_start_time = scaled_flow.time_index[i]
                time_index = pd.date_range(start=pred_start_time, periods=forecast_horizon,
                                           freq=scaled_flow.freq)

                forecast_ts = TimeSeries.from_times_and_values(time_index, pred_tensor.cpu().numpy())
                all_forecasts.append(forecast_ts)

            backtest_scaled = concatenate(all_forecasts)
            simul_values_denorm = self.series.target_scaler.inverse_transform(backtest_scaled.values())
            simulation = TimeSeries.from_times_and_values(backtest_scaled.time_index, simul_values_denorm)
            return simulation

        elif model_type in ['nbeats', 'nhits']:

            # --- NOVO BLOCO: Backtesting com historical_forecasts ---
            start_point = self.model.input_chunk_length
            # O backtest deve ser executado nos dados normalizados, que foi como o modelo treinou
            backtest_scaled = self.model.historical_forecasts(
                series=scaled_flow,  # Série completa normalizada
                past_covariates=scaled_covariates,  # Covariáveis completas normalizadas
                start=start_point,  # Começa a prever DEPOIS do treino
                forecast_horizon=1,  # Gera uma previsão de 1 passo de cada vez (mais rigoroso)
                stride=1,  # Avança de 1 em 1 dia
                retrain=False,  # Usa o modelo já treinado, não retreina a cada passo (muito mais rápido)
                verbose=False
            )

            simul_values_denorm = self.series.target_scaler.inverse_transform(backtest_scaled.values())
            simulation = TimeSeries.from_times_and_values(backtest_scaled.time_index, simul_values_denorm)
            return simulation

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
    def __init__(self, params, configs, series, preds=None, simul=None, losses=None, models=None, residuals=None, all_params=None):
        super().__init__(params, configs, series, preds, simul, losses, models, residuals, all_params)
        self.title("LHC - Executando Modelo")
        self.centralize_window()

        self.all_params["LHC"] = self.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tensor sets baseados em GetSeries (prep treino e validação)
        self.train_target = torch.tensor(self.series.scaled_train_target.values(), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.valid_target = torch.tensor(self.series.scaled_valid_target.values(), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.train_cov = torch.tensor(self.series.scaled_train_cov.values(), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.valid_cov = torch.tensor(self.series.scaled_valid_cov.values(), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.prate_covariates = torch.tensor(self.series.scaled_prate_covariates.values(), dtype=torch.float32).unsqueeze(0).to(self.device)

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

        self.after(1000, self.post_training)

    def post_training(self):
        n_predict = len(self.valid_target.squeeze(0))
        # Prepara a janela de aquecimento (warm-up) com os últimos dados de treino
        train_len = self.train_target.shape[1]
        warm_up_flow = self.train_target[:, train_len - self.input_len:, :]
        warm_up_cov = self.train_cov[:, train_len - self.input_len:, :]
        initial_input_seq_pred = torch.cat([warm_up_flow, warm_up_cov], dim=2)

        # As covariáveis futuras para a predição são as do conjunto de validação
        future_covariates_pred = self.valid_cov

        preds_tensor_scaled = recursive_predict(
            self.model,
            initial_input_seq_pred,
            future_covariates_pred,
            n_predict,
            self.device
        )
        preds_numpy_denorm = self.series.target_scaler.inverse_transform(preds_tensor_scaled.cpu().numpy())
        preds_ts = TimeSeries.from_times_and_values(self.pred_time_index, preds_numpy_denorm)
        self.predictions["LHC"] = preds_ts
        self.label_progress.configure(text="Processando os resíduos. Aguarde.")

        residuals_ts = residuals_timeseries(self.series.valid_target, preds_ts)
        self.residuals["LHC"] = residuals_ts

        simulation_ts = self.backtest_simulation(self.model, 'lhc')
        self.simulations["LHC"] = simulation_ts

        self.losses["LHC"] = self.loss_tracker.losses
        self.models["LHC"] = self.model
        self.training_finished()

        self.after(1000, self.next_model_run)

class ModelRunDartsWindow(ModelRunWindow):
    def __init__(self, params, configs, series, model_class, model_name, preds=None, simul=None, losses=None,
                 models=None, residuals=None, all_params=None):
        super().__init__(params, configs, series, preds, simul, losses, models, residuals,all_params)
        self.title(f"{model_name} - Executando Modelo")
        self.centralize_window()
        self.model_class = model_class  # Ex: NBEATSModel ou NHiTSModel
        self.model_name = model_name  # Ex: "N-BEATS" ou "N-HiTS"
        self.all_params[self.model_name] = self.params
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
        self.model.fit(series=self.series.scaled_train_target,
                       past_covariates=self.series.scaled_train_cov,
                       val_series=self.series.scaled_valid_target,
                       val_past_covariates=self.series.scaled_valid_cov)

        # Após o fim do treinamento, agenda a execução do pós-treino na thread principal da GUI
        self.after(1000, self.post_training)

    def post_training(self):

        if self.params.get('save_checkpoints', 'false') == 'true':
            self.model = self.model_class.load_from_checkpoint(self.model.model_name)

        # Predição
        predictions_scaled = self.model.predict(series=self.series.scaled_train_target,
                                         past_covariates=self.series.scaled_prate_covariates,
                                         n=len(self.series.scaled_valid_target))

        pred_values_denorm = self.series.target_scaler.inverse_transform(predictions_scaled.values())
        predictions = TimeSeries.from_times_and_values(predictions_scaled.time_index, pred_values_denorm)
        self.predictions[self.model_name] = predictions

        # Simulação
        simulation = self.backtest_simulation(self.model, self.model_name.lower().replace('-', ''))
        self.simulations[self.model_name] = simulation

        # Resíduos
        residuals_scaled = self.model.residuals(self.series.scaled_valid_target, past_covariates=self.series.scaled_valid_cov,
                                                verbose=False, retrain=False)
        residual_values_denorm = residuals_scaled.values() * self.series.target_scaler.scale_[0]
        residuals = TimeSeries.from_times_and_values(residuals_scaled.time_index, residual_values_denorm)
        self.residuals[self.model_name] = residuals

        # Salva resultados e finaliza
        self.losses[self.model_name] = self.loss_tracker.losses
        self.models[self.model_name] = self.model
        self.training_finished()

        self.after(1000, self.next_model_run)

class ModelRunNBEATSWindow(ModelRunDartsWindow):
    def __init__(self, params, configs, series, preds=None, simul=None, losses=None, models=None, residuals=None, all_params=None):
        super().__init__(params=params,
                         configs=configs,
                         series=series,
                         model_class=NBEATSModel,
                         model_name="N-BEATS",
                         preds=preds,
                         simul=simul,
                         losses=losses,
                         models=models,
                         residuals=residuals,
                         all_params=all_params)

class ModelRunNHiTSWindow(ModelRunDartsWindow):
    def __init__(self, params, configs, series, preds=None, simul=None, losses=None, models=None, residuals=None, all_params=None):
        super().__init__(params=params,
                         configs=configs,
                         series=series,
                         model_class=NHiTSModel,
                         model_name="N-HiTS",
                         preds=preds,
                         simul=simul,
                         losses=losses,
                         models=models,
                         residuals=residuals,
                         all_params=all_params)