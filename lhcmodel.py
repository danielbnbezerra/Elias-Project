import torch

import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from darts import TimeSeries
from pytorch_lightning.callbacks import Callback


class LSTMModel(nn.Module):
    """Arquitetura LSTM básica usada internamente pelo LHCModel."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class LHCModel(pl.LightningModule):
    def __init__(self, params, input_size):
        super().__init__()

        self.save_hyperparameters(params)  # salva params para checkpoint

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            output_size=1,
            dropout=params.get('dropout', 0.0)
        )
        self.loss_fn = nn.MSELoss()
        self.learning_rate = params.get('learning_rate', 0.001)
        self.random_state = params.get('random_state', 42)
        torch.manual_seed(self.random_state)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y[:, -1, :])
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y[:, -1, :])
        # Loga a loss de validação
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def series_to_batches(series_list, input_len, output_len):
    """Transforma séries em janelas (batches) de entrada/target para treino."""
    X, Y = [], []
    for series in series_list:
        arr = series.squeeze(0).numpy()
        seq_len, n_features = arr.shape
        max_start = seq_len - input_len - output_len + 1
        for start in range(max_start):
            x = arr[start : start + input_len]
            y = arr[start + input_len : start + input_len + output_len, 0:1]
            X.append(x)
            Y.append(y)
    inputs = torch.tensor(np.array(X, dtype=np.float32))
    targets = torch.tensor(np.array(Y, dtype=np.float32))
    return inputs, targets


def recursive_predict(lit_model, initial_input_seq, future_covariates, n_predict, device):
    """
    Realiza predição/simulação recursiva de forma robusta.

    Args:
        lit_model: O modelo treinado.
        initial_input_seq: Tensor de warm-up com formato [1, input_len, n_features], já contendo vazão e covariáveis alinhadas.
        future_covariates: Tensor com as covariáveis futuras conhecidas para o período de predição.
        n_predict: Número de passos a prever.
        device: Dispositivo (cpu ou cuda).
    """
    lit_model.model.eval()
    predictions = []
    input_seq = initial_input_seq.to(device)
    num_covariates = future_covariates.shape[2]

    with torch.no_grad():
        for i in range(n_predict):
            out = lit_model.model(input_seq)  # Shape: [1, 1]
            predictions.append(out.cpu().numpy())

            # Obtém a covariável para o próximo passo
            if i < future_covariates.shape[1]:
                next_cov = future_covariates[:, i:i + 1, :]
            else:
                # Se acabarem as covariáveis futuras, preenche com zeros
                next_cov = torch.zeros(1, 1, num_covariates).to(device)

            # Cria o vetor de features para o próximo passo (vazão prevista + covariável futura)
            next_input = torch.cat([out.unsqueeze(2), next_cov], dim=2)

            # Desliza a janela de entrada: remove o passo mais antigo e adiciona o novo
            input_seq = torch.cat([input_seq[:, 1:, :], next_input], dim=1)

    predictions = np.concatenate(predictions, axis=0)
    return torch.tensor(predictions)

def predictions_to_timeseries(predictions_tensor, time_index):
    preds_array = predictions_tensor.squeeze(1).cpu().numpy()
    return TimeSeries.from_times_and_values(time_index, preds_array)

def residuals_timeseries(valid_target_ts, predictions_ts):
    """Calcula resíduos como série temporal no formato Darts, alinhando as séries."""
    return valid_target_ts - predictions_ts
