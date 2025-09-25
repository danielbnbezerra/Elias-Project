import math
import os
import platform
import subprocess
import tempfile
import torch

import tkinter as tk
import customtkinter as ctk
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tkinter import messagebox
from darts.utils.statistics import plot_residuals_analysis, plot_acf, plot_pacf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from metrics import *


class PlotWindow(ctk.CTkToplevel):
    def __init__(self, series, predictions, simulations, residuals, losses, models, all_params):
        super().__init__()
        self.title("Visualização de Resultados")
        self.grab_set()
        self.create_submenu()
        self.timeseries = series
        self.all_params = all_params

        self.series = {"Precipitação":series.prate,
                       "Precipitação Acumulada 3 dias": series.prate_t_minus_3,
                       "Precipitação Acumulada 5 dias": series.prate_t_minus_5,
                       "Vazão":series.flow}

        self.predictions = predictions
        self.simulations = simulations
        self.residuals = residuals
        self.losses = losses
        self.models = models

        # Menu lateral
        self.menu_frame = ctk.CTkFrame(self, width=200, fg_color='#DDDDDD')
        self.menu_frame.pack(side="left", fill="both")

        new_graph_button = ctk.CTkButton(self.menu_frame, text="Novo Gráfico", command=self.open_new_graph_modal)
        new_graph_button.pack(pady=(20,10), padx=20)

        delete_graph_button = ctk.CTkButton(self.menu_frame, text="Excluir Gráficos", command=self.open_delete_graphs_modal)
        delete_graph_button.pack(pady=(0, 20), padx=20)

        clear_all_button = ctk.CTkButton(self.menu_frame, text="Limpar todos os gráficos", command=self.clear_all_graphs)
        clear_all_button.pack(pady=(0, 20), padx=20)

        self.graphs = []  # Cada item: dict {frame, canvas, name}

        self.show_initial_graph_area()
        self.centralize_window()
        self.state("zoomed")

    def create_submenu(self):
        #Export Menu
        export_menu = tk.Menu(self)
        self.config(menu=export_menu)

        #Report Menu
        report_menu = tk.Menu(export_menu, tearoff=0)
        export_menu.add_cascade(label="Resultados", menu=report_menu)
        report_menu.add_command(label="Gerar Relatório", command=self.generate_report_pdf)
        report_menu.add_command(label="Exportar Gráficos", command=self.export_graphs_to_pdf)
        report_menu.add_command(label="Exportar Modelos", command=self.export_models)
        report_menu.add_separator()
        report_menu.add_command(label="Sair", command=self.destroy)

    def show_initial_graph_area(self):
        if hasattr(self, 'graph_area_frame'):
            self.graph_area_frame.destroy()
        self.graph_area_frame = ctk.CTkFrame(self, fg_color='#EBEBEB')
        self.graph_area_frame.pack(side="right", fill="both", expand=True)

        self.initial_label = ctk.CTkLabel(
            self.graph_area_frame,
            text="Clique em 'Novo Gráfico' para adicionar gráficos",
            font=ctk.CTkFont(size=18)
        )
        self.initial_label.place(relx=0.5, rely=0.5, anchor="center")

    def open_new_graph_modal(self):
        CreateGraphModal(
            self,
            self.series,
            self.predictions,
            self.simulations,
            self.losses,
            self.residuals,
            self.add_graph
        )

    def open_delete_graphs_modal(self):
        if not self.graphs:
            return
        names = [g["name"] for g in self.graphs]
        DeleteGraphsModal(self, names, self.delete_graphs)

    def add_graph(self, name_entry, selected_data_indices):
        if hasattr(self, "initial_label") and self.initial_label.winfo_exists():
            self.graph_area_frame.destroy()
            self.graph_area_frame = ctk.CTkFrame(self, fg_color='#EBEBEB')
            self.graph_area_frame.pack(side="right", fill="both", expand=True)

            self.scrollable_graph_frame = ctk.CTkScrollableFrame(self.graph_area_frame, fg_color='#EBEBEB')
            self.scrollable_graph_frame.pack(fill="both", expand=True)

            self.graphs = []

        frame = ctk.CTkFrame(self.scrollable_graph_frame, corner_radius=10)

        fig = plt.figure(figsize=(20,8))
        ax = fig.add_subplot(111)

        # Séries
        if selected_data_indices.get("Séries"):
            for name, series_obj in selected_data_indices["Séries"].items():
                ax.plot(series_obj.time_index, series_obj.values(), label=name)
            series_size = len(self.series["Vazão"])//365
            if series_size > 5:
                ticks = pd.date_range(start=self.series["Vazão"].start_time(), end=self.series["Vazão"].end_time(), freq='YS')
                fmt = '%Y'
            else:
                ticks = pd.date_range(start=self.series["Vazão"].start_time(), end=self.series["Vazão"].end_time(),
                                      freq='MS')
                fmt = '%m/%y'
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            ax.tick_params(axis='x', labelsize=22, rotation=45)
            ax.tick_params(axis='y', labelsize=22)
            ax.set_xlabel('Tempo', fontsize=20)
            ax.legend(fontsize=22, loc='best')
            ax.grid(True)
            ax.set_title(name_entry, fontsize=22)

        # Previsões (sempre plota flow para comparação)
        elif selected_data_indices.get("Previsões"):
            # Plota série flow inteira com tempo original para comparação
            ax.plot(self.series["Vazão"].time_index, self.series["Vazão"].values(), label="Série Observada")

            # Para cada previsão, usa seu time_index para alinhamento correto no tempo
            for name, pred_series in selected_data_indices["Previsões"].items():
                ax.plot(pred_series.time_index, pred_series.values(), label=f"{name} - Previsão")
            series_size = len(self.series["Vazão"]) // 365
            if series_size > 5:
                ticks = pd.date_range(start=self.series["Vazão"].start_time(), end=self.series["Vazão"].end_time(),
                                  freq='YS')
                fmt = '%Y'
            else:
                ticks = pd.date_range(start=self.series["Vazão"].start_time(), end=self.series["Vazão"].end_time(),
                                  freq='MS')
                fmt = '%m/%y'
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            ax.tick_params(axis='x', labelsize=22, rotation=45)
            ax.tick_params(axis='y', labelsize=22)
            ax.set_xlabel('Tempo', fontsize=20)
            ax.set_ylabel('Vazão', fontsize=20)
            ax.legend(fontsize=22, loc='best')
            ax.grid(True)
            ax.set_title(name_entry, fontsize=22)

        # Simulações
        elif selected_data_indices.get("Simulações"):
            common_index = self.series["Vazão"].time_index
            for simul_series in selected_data_indices["Simulações"].values():
                common_index = common_index.intersection(simul_series.time_index)
            size_series = len(common_index)//365
            # Plota série flow inteira com tempo original para comparação
            flow_aligned = self.series["Vazão"][common_index]
            ax.plot(flow_aligned.time_index, flow_aligned.values(), label="Série Observada")

            # Para cada previsão, usa seu time_index para alinhamento correto no tempo
            for name, simul_series in selected_data_indices["Simulações"].items():
                simul_series_aligned = simul_series[common_index]
                ax.plot(simul_series_aligned.time_index, simul_series_aligned.values(), label=f"{name} - Simulação")
            if size_series > 5:
                ticks = pd.date_range(start=flow_aligned.start_time(), end=flow_aligned.end_time(), freq='YS')
                fmt = '%Y'
            else:
                ticks = pd.date_range(start=flow_aligned.start_time(), end=flow_aligned.end_time(), freq='MS')
                fmt = '%m/%y'
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            ax.tick_params(axis='x', labelsize=22, rotation=45)
            ax.tick_params(axis='y', labelsize=22)
            ax.set_xlabel('Tempo', fontsize=20)
            ax.set_ylabel('Vazão', fontsize=20)
            ax.legend(fontsize=22, loc='best')
            ax.grid(True)
            ax.set_title(name_entry, fontsize=22)

        # Curvas de Aprendizado
        elif selected_data_indices.get("Curvas de Aprendizado"):
            for name, loss_values in selected_data_indices["Curvas de Aprendizado"].items():
                ax.plot(range(len(loss_values)), loss_values, label=f"{name} - Curva de Aprendizado")
            ax.set_xlabel("Épocas", fontsize=20)
            ax.set_ylabel("Perdas", fontsize=20)
            ax.tick_params(axis='x', labelsize=22, rotation=45)
            ax.tick_params(axis='y', labelsize=22)
            ax.legend(fontsize=22, loc='best')
            ax.grid(True)
            ax.set_title(name_entry, fontsize=22)


        # Resíduos (único selecionado)
        elif selected_data_indices.get("Resíduos"):
            for name, res_obj in selected_data_indices["Resíduos"].items():
                plt.close('all')
                plot_residuals_analysis(res_obj)
                fig = plt.gcf()
                fig.set_size_inches(8, 6)
                fig.suptitle(f"{name} - Resíduos", fontsize=22)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=(0,5), pady=(0,5))
                self.graphs.append({"frame": frame, "canvas": canvas, "name": name_entry})
                self.update_grid_layout()
                return

        if not ax.has_data():
            frame.destroy()
            return

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=(0,5), pady=(0,5))

        self.graphs.append({"frame": frame, "canvas": canvas, "name": name_entry})
        self.update_grid_layout()

    def delete_graphs(self, names_to_delete):
        to_remove = [g for g in self.graphs if g["name"] in names_to_delete]
        for g in to_remove:
            g["frame"].destroy()
            self.graphs.remove(g)
        self.update_grid_layout()
        if len(self.graphs) == 0:
            self.show_initial_graph_area()

    def clear_all_graphs(self):
        for g in self.graphs:
            g["frame"].destroy()
        self.graphs.clear()
        self.show_initial_graph_area()

    def update_grid_layout(self):
        if not hasattr(self, 'scrollable_graph_frame'):
            return
        total = len(self.graphs)
        if total == 0:
            return

        cols=1
        rows = math.ceil(total / cols)

        for r in range(rows):
            self.scrollable_graph_frame.grid_rowconfigure(r, weight=1)
        for c in range(cols):
            self.scrollable_graph_frame.grid_columnconfigure(c, weight=1)

        for graph in self.graphs:
            graph["frame"].grid_forget()

        for idx, graph in enumerate(self.graphs):
            r = idx // cols
            c = idx % cols
            graph["frame"].grid(row=r, column=c, sticky="nsew", padx=5, pady=5)

    def centralize_window(self):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+{0}+{0}")

    def export_models(self):
        """
        Exporta cada modelo em self.models para arquivos .py, incluindo os imports necessários
        para rodar cada modelo de forma independente, além dos parâmetros embutidos e funções
        avançadas como backtesting.
        """
        try:
            for name, model in self.models.items():
                base_name = name.lower()
                weights_filename = f"{base_name}_weights.pth"
                ckpt_filename = f"{base_name}_weights.pth.ckpt"
                script_filename = f"{base_name}_model.py"

                if name == "LHC":
                    # Salvar pesos do modelo LHC
                    torch.save(model.state_dict(), weights_filename)

                    # Parâmetros usados no modelo
                    params = self.all_params[name]
                    params_repr = repr(params)

                    # Script completo para o LHC, incluindo imports e a nova função de backtest
                    lhc_complete_code = f"""import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from darts import TimeSeries, concatenate

# --- Parâmetros e Definição do Modelo ---

# Parâmetros usados no treinamento original
params = {params_repr}

class LSTMModel(nn.Module):
    \"\"\"Arquitetura LSTM básica usada internamente pelo LHCModel.\"\"\"
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
    \"\"\"Classe LightningModule para o LHCModel.\"\"\"
    def __init__(self, params, input_size):
        super().__init__()
        self.save_hyperparameters(params)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# --- Funções de Predição e Avaliação ---

def recursive_predict(lit_model, initial_input_seq, future_covariates, n_predict, device):
    \"\"\"
    Realiza predição/simulação recursiva de forma robusta.
    Esta é a função central de previsão usada tanto para forecasting quanto para backtesting.
    \"\"\"
    lit_model.model.eval()
    predictions = []
    input_seq = initial_input_seq.to(device)
    num_covariates = future_covariates.shape[2]

    with torch.no_grad():
        for i in range(n_predict):
            out = lit_model.model(input_seq)  # Shape: [1, 1]
            predictions.append(out.cpu().numpy())

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

def backtest_simulation_lhc(model, scaled_flow_series, scaled_covariates_series, forecast_horizon=1, stride=1):
    \"\"\"
    Executa um backtesting manual para o modelo LHC, simulando historical_forecasts.
    Esta função desliza uma janela sobre a série temporal, fazendo previsões passo a passo
    para simular como o modelo performaria em dados nunca vistos.

    Args:
        model: O modelo LHC treinado e carregado.
        scaled_flow_series (TimeSeries): A série de vazão completa, já normalizada.
        scaled_covariates_series (TimeSeries): As covariáveis completas, já normalizadas.
        forecast_horizon (int): O número de passos a prever em cada iteração (default=1).
        stride (int): O número de passos para avançar a janela a cada iteração (default=1).

    Returns:
        TimeSeries: Uma única série temporal com as previsões do backtest.
                    Atenção: Os valores retornados estão na escala normalizada,
                    exigindo transformação inversa para análise.
    \"\"\"
    input_len = model.hparams.input_length
    device = next(model.parameters()).device
    all_forecasts = []

    # O ponto de início é o índice após o primeiro período de aquecimento completo
    start_index = input_len

    # Loop que "desliza" pelo período de validação/teste
    for i in range(start_index, len(scaled_flow_series) - forecast_horizon + 1, stride):
        # 1. Pega a janela de "aquecimento" (warm-up) que antecede o ponto da previsão
        history_start_index = i - input_len
        history_end_index = i

        warm_up_flow_tensor = torch.tensor(scaled_flow_series[history_start_index:history_end_index].values(),
                                           dtype=torch.float32).unsqueeze(0)
        warm_up_cov_tensor = torch.tensor(
            scaled_covariates_series[history_start_index:history_end_index].values(),
            dtype=torch.float32).unsqueeze(0)
        initial_input_seq = torch.cat([warm_up_flow_tensor, warm_up_cov_tensor], dim=2)

        # 2. Pega as covariáveis futuras conhecidas para o horizonte da previsão
        future_cov_start_index = i
        future_cov_end_index = i + forecast_horizon
        future_covariates_tensor = torch.tensor(
            scaled_covariates_series[future_cov_start_index:future_cov_end_index].values(),
            dtype=torch.float32).unsqueeze(0)

        # 3. Chama a função de previsão recursiva
        pred_tensor = recursive_predict(
            model,
            initial_input_seq.to(device),
            future_covariates_tensor.to(device),
            n_predict=forecast_horizon,
            device=device
        )

        # 4. Armazena o resultado com o timestamp correto
        pred_start_time = scaled_flow_series.time_index[i]
        time_index = pd.date_range(start=pred_start_time, periods=forecast_horizon,
                                   freq=scaled_flow_series.freq_str)

        forecast_ts = TimeSeries.from_times_and_values(time_index, pred_tensor.cpu().numpy())
        all_forecasts.append(forecast_ts)

    backtest_scaled = concatenate(all_forecasts)
    return backtest_scaled

def predictions_to_timeseries(predictions_tensor, time_index):
    \"\"\"Converte tensor de predições para TimeSeries do Darts.\"\"\"
    preds_array = predictions_tensor.squeeze(1).cpu().numpy()
    return TimeSeries.from_times_and_values(time_index, preds_array)

# --- Funções Utilitárias para Carregamento ---

def load_model(weights_path='{weights_filename}', input_size=None):
    \"\"\"
    Carrega o modelo LHC a partir dos pesos salvos.

    Args:
        weights_path: Caminho para o arquivo de pesos (.pth)
        input_size: Tamanho da dimensão de entrada (número de features, ex: 3 para vazão + 2 covariáveis)

    Returns:
        Modelo LHC carregado e pronto para inferência
    \"\"\"
    if input_size is None:
        raise ValueError("input_size deve ser fornecido para carregar o modelo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A input_size é inferida a partir dos dados, aqui precisa ser explícita
    model = LHCModel(params, input_size=input_size) 
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    print("Modelo LHC - Arquivo de exportação")
    print(f"Parâmetros usados: {{params}}")
    print("\\nFunções disponíveis:")
    print(" - load_model(weights_path, input_size): Carrega o modelo treinado.")
    print(" - recursive_predict(...): Realiza uma única previsão para o futuro.")
    print(" - backtest_simulation_lhc(model, ...): Executa uma simulação histórica completa.")
    # Exemplo de como usar:
    # 1. Carregue suas séries temporais (flow, covariates) com Darts e normalize-as.
    # 2. n_features = series_covariates.width + 1
    # 3. lhc = load_model(input_size=n_features)
    # 4. backtest_results = backtest_simulation_lhc(lhc, scaled_flow, scaled_covariates)
    # 5. Lembre-se de aplicar a transformação inversa (scaler.inverse_transform) nos resultados.
"""

                    with open(script_filename, "w", encoding="utf-8") as f:
                        f.write(lhc_complete_code)

                elif name in ("N-BEATS", "N-HiTS"):
                    # Salvar modelo DARTS
                    model.save(weights_filename)

                    # Parâmetros usados
                    params = self.all_params[name]
                    params_repr = repr(params)

                    # Importações necessárias para modelos DARTS
                    load_code = f"""import torch
from darts.models import {type(model).__name__}

# Parâmetros usados no treinamento
params = {params_repr}

def load_model(weights_path='{weights_filename}'):
    \"\"\"
    Carrega o modelo {name} a partir do arquivo salvo.

    Args:
        weights_path: Caminho para o arquivo do modelo

    Returns:
        Modelo {name} carregado e pronto para uso
    \"\"\"
    # O método .load da Darts carrega a arquitetura e os pesos.
    model = {type(model).__name__}.load(weights_path)
    return model

if __name__ == "__main__":
    print("Modelo {name} - Arquivo de exportação")
    print(f"Parâmetros usados: {{params}}")
    print("Use load_model() para carregar o modelo treinado")
    print("Modelos Darts possuem o método .historical_forecasts() para backtesting.")
"""

                    with open(script_filename, "w", encoding="utf-8") as f:
                        f.write(load_code)

                else:
                    print(f"Exportação para modelo {name} não implementada.")

                if os.path.exists(ckpt_filename):
                    os.remove(ckpt_filename)

            messagebox.showinfo("Exportação Concluída",
                                "Modelos exportados com sucesso!",
                                parent=self)
        except Exception as e:
            messagebox.showerror("Erro na Exportação",
                                 f"Ocorreu um erro ao exportar os modelos: {e}",
                                 parent=self)



    def generate_report_pdf(self, name_file=f"Relatório Completo"):
        filename = self.get_unique_filename(name_file,"pdf")
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        margin = 50
        line_height = 14

        y_position = height - margin

        for idx, (model_name, predicted_ts) in enumerate(self.predictions.items()):
            actual_ts = self.timeseries.valid_target
            original_ts= self.timeseries.flow
            simul_ts = self.simulations[model_name]

            original = original_ts.values().flatten()
            simul = simul_ts.values().flatten()

            rmse_value = rmse(actual_ts, predicted_ts)
            mae_value = mae(actual_ts, predicted_ts)
            nse_value = nse(actual_ts, predicted_ts)
            kge_value = kge(actual_ts, predicted_ts)

            mk_result_original = mann_kendall_test(original)
            mk_result_simul = mann_kendall_test(simul)

            dagostino_result = dagostino_k_squared_test(self.residuals[model_name])
            anderson_result = anderson_darling_test(self.residuals[model_name])
            shapiro_result = shapiro_wilk_test(self.residuals[model_name])

            adf_original = adf_test(original)
            kpss_original = kpss_test(original)
            adf_simul = adf_test(simul)
            kpss_simul = kpss_test(simul)

            #Gerar da série de entrada para comparar
            # Gerar gráficos ACF e PACF e salvar imagens temporárias
            with tempfile.TemporaryDirectory() as tmpdir:
                acf_path_original = os.path.join(tmpdir, "acf_original.png")
                pacf_path_original = os.path.join(tmpdir, "pacf_original.png")
                acf_path_simul = os.path.join(tmpdir, "acf_simul.png")
                pacf_path_simul = os.path.join(tmpdir, "pacf_simul.png")

                plt.figure(figsize=(6, 3))
                plot_acf(original_ts, max_lag=min(40, int(len(original_ts)/4)))
                plt.suptitle("Autocorrelação Observação", fontsize=20)
                plt.tight_layout()
                plt.savefig(acf_path_original, dpi=300, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(6, 3))
                plot_acf(simul_ts, max_lag=min(40, int(len(simul_ts) / 4)))
                plt.suptitle("Autocorrelação Simulação", fontsize=20)
                plt.tight_layout()
                plt.savefig(acf_path_simul, dpi=300, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(6, 3))
                plot_pacf(original_ts, max_lag=min(40, int(len(original_ts)/4)))
                plt.suptitle("Autocorrelação Parcial Observação", fontsize=20)
                plt.tight_layout()
                plt.savefig(pacf_path_original, dpi=300, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(6, 3))
                plot_pacf(simul_ts, max_lag=min(40, int(len(simul_ts) / 4)))
                plt.suptitle("Autocorrelação Parcial Simulação", fontsize=20)
                plt.tight_layout()
                plt.savefig(pacf_path_simul, dpi=300, bbox_inches='tight')
                plt.close()

                # Escrever texto no PDF
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, y_position, f"Modelo: {model_name}")
                y_position -= line_height * 2

                c.setFont("Helvetica", 12)
                lines = [
                    f"Métricas:",
                    f"",
                    f"RMSE: {rmse_value:.4f}",
                    f"MAE: {mae_value:.4f}",
                    f"NSE: {nse_value:.4f}",
                    f"KGE: {kge_value:.4f}",
                    f"",
                    f"",
                    f"Normalidade dos Resíduos:",
                    f"",
                    f" - D’Agostino: {dagostino_result['result']} (p-value={dagostino_result.get('p_value', 'N/A'):.4f})",
                    f" - Anderson-Darling: {anderson_result['result']} (Estatística={anderson_result.get('statistic', 'N/A'):.4f}, Critério 5%={anderson_result.get('critical_value_5pct', 'N/A'):.4f})",
                    f" - Shapiro-Wilk: {shapiro_result['result']} (p-value={shapiro_result.get('p_value', 'N/A'):.4f})",
                    f"",
                    f"",
                    f"Tendência (Mann-Kendall):",
                    f"",
                    f"Série Observada:",
                    f"Resultado: {mk_result_original['trend']}, Tau: {mk_result_original['tau']:.4f}, p-value: {mk_result_original['p']:.4f}",
                    f"",
                    f"Simulação:",
                    f"Resultado: {mk_result_simul['trend']}, Tau: {mk_result_simul['tau']:.4f}, p-value: {mk_result_simul['p']:.4f}",
                    f"",
                    f"",
                    f"Estacionariedade:",
                    f"",
                    f"ADF:",
                    f" - Observação: {adf_original['result']} (Estatística={adf_original['adf_statistic']:.4f}, p-value={adf_original['p_value']:.4f})",
                    f" - Simulação: {adf_simul['result']} (Estatística={adf_simul['adf_statistic']:.4f}, p-value={adf_simul['p_value']:.4f})",
                    f"",
                    f"KPSS:",
                    f" - Observação: {kpss_original['result']} (Estatística={kpss_original['kpss_statistic']:.4f}, p-value={kpss_original['p_value']:.4f})",
                    f" - Simulação: {kpss_simul['result']} (Estatística={kpss_simul['kpss_statistic']:.4f}, p-value={kpss_simul['p_value']:.4f})",
                ]

                for line in lines:
                    if y_position < margin:
                        c.showPage()
                        y_position = height - margin
                        c.setFont("Helvetica", 12)
                    c.drawString(margin, y_position, line)
                    y_position -= line_height

                # Inserir imagens dos gráficos
                if y_position < 400:  # Precisa de mais espaço para a grade
                    c.showPage()
                    y_position = height - margin

                    # Carrega as 4 imagens
                acf_img_original = ImageReader(acf_path_original)
                acf_img_simul = ImageReader(acf_path_simul)
                pacf_img_original = ImageReader(pacf_path_original)
                pacf_img_simul = ImageReader(pacf_path_simul)

                # Define as dimensões e posições da grade
                plot_width = 240
                plot_height = 160
                h_spacing = 30
                v_spacing = 20

                x1 = margin
                x2 = margin + plot_width + h_spacing

                # Posição Y da primeira linha de gráficos
                y_top_row = y_position - plot_height - 10
                # Posição Y da segunda linha de gráficos
                y_bottom_row = y_top_row - plot_height - v_spacing

                # Desenha a primeira linha: [ACF Original] [ACF Simulada]
                c.drawImage(acf_img_original, x1, y_top_row, width=plot_width, height=plot_height)
                c.drawImage(acf_img_simul, x2, y_top_row, width=plot_width, height=plot_height)

                # Desenha a segunda linha: [PACF Original] [PACF Simulada]
                c.drawImage(pacf_img_original, x1, y_bottom_row, width=plot_width, height=plot_height)
                c.drawImage(pacf_img_simul, x2, y_bottom_row, width=plot_width, height=plot_height)

                # Atualiza a posição Y para o conteúdo seguinte
                y_position = y_bottom_row - 20

                # Linha separadora
                c.line(margin, y_position, width - margin, y_position)
                y_position -= line_height

                if idx < len(self.predictions) - 1:
                    c.showPage()
                    y_position = height - margin

        c.save()
        self.open_pdf_file(filename)

    def get_unique_filename(self, base_name, extension):
        filename = f"{base_name}.{extension}"
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_name} ({counter}).{extension}"
            counter += 1
        return filename

    # Função para salvar figures matplotlib em imagens temporárias
    def save_figure_temp(self, fig):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight', dpi=300)
            return tmpfile.name

    # Função que cria um PDF apenas com os gráficos criados
    # Espera receber a lista self.graphs da classe PlotWindow
    # Supondo que esta função esteja dentro de uma classe, como no seu exemplo original.
    # Se não estiver, remova o 'self'.
    def export_graphs_to_pdf(self, name_file='Gráficos Exportados'):
        """
        Exporta uma lista de gráficos para um arquivo PDF, com cada gráfico
        em uma página separada e em modo paisagem.
        """
        filename = self.get_unique_filename(name_file, 'pdf')

        # 1. Alteração: Definir a página para o modo paisagem (landscape)
        c = canvas.Canvas(filename, pagesize=landscape(letter))
        width, height = landscape(letter)  # Agora 'width' é a dimensão maior

        margin = 50  # Um pouco mais de margem para um visual mais limpo

        if not self.graphs:
            c.drawString(margin, height - margin, "Nenhum gráfico para exportar.")
            c.save()
            return filename

        for i, graph in enumerate(self.graphs):
            # --- Configuração do Título da Página ---
            c.setFont("Helvetica-Bold", 16)
            # Centraliza o título no topo da página
            title = f"Gráfico {i + 1}: {graph.get('name', '')}"
            c.drawCentredString(width / 2, height - margin, title)

            # --- Preparação da Imagem do Gráfico ---
            fig = graph['canvas'].figure
            img_path = self.save_figure_temp(fig)

            # --- Cálculo de Dimensões para a Imagem ---
            # Área disponível para o gráfico (considerando margens e espaço para o título)
            available_width = width - (2 * margin)
            available_height = height - (2 * margin) - 40  # Subtrai espaço extra para o título

            # Obtém dimensões originais para manter a proporção
            fig_width, fig_height = fig.get_size_inches()
            dpi = fig.get_dpi()
            original_img_width = fig_width * dpi
            original_img_height = fig_height * dpi

            # Calcula o fator de escala para que a imagem caiba na área disponível
            scale_factor = min(available_width / original_img_width, available_height / original_img_height)

            display_width = original_img_width * scale_factor
            display_height = original_img_height * scale_factor

            # --- Posicionamento e Desenho da Imagem ---
            # Calcula a posição (x, y) para centralizar a imagem na página
            x_pos = (width - display_width) / 2
            y_pos = (height - display_height) / 2

            c.drawImage(img_path, x_pos, y_pos, width=display_width, height=display_height)

            # Remove o arquivo de imagem temporário
            os.remove(img_path)

            # 2. Alteração: Finaliza a página atual e avança para a próxima
            c.showPage()

        # Salva o arquivo PDF final
        c.save()
        self.open_pdf_file(filename)

    def open_pdf_file(self, path):
        if platform.system() == 'Windows':
            os.startfile(path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', path])
        else:  # Linux e outros
            subprocess.call(['xdg-open', path])

class CreateGraphModal(ctk.CTkToplevel):
    def __init__(self, parent, series, predictions, simulations, losses, residuals, callback):
        super().__init__(parent)
        self.title("Novo Gráfico")
        self.grab_set()
        self.centralize_window()

        self.callback = callback

        self.series_names = list(series.keys())
        self.series = series
        self.predictions = predictions
        self.simulations = simulations
        self.losses = losses
        self.residuals = residuals

        ctk.CTkLabel(self, text="Nome do gráfico:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,0))
        self.name_entry = ctk.CTkEntry(self)
        self.name_entry.pack(pady=(0,15), padx=20, fill="x")
        self.name_entry.focus()

        ctk.CTkLabel(self, text="Tipo de gráfico:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(2,5))
        self.graph_type = ctk.StringVar(value="Séries")

        self.options_graph_frame = ctk.CTkFrame(self, fg_color='transparent')
        self.options_graph_frame.pack(pady=5)
        self.rb_series = ctk.CTkRadioButton(self.options_graph_frame, text="Séries", variable=self.graph_type, value="Séries", command=self.update_options)
        self.rb_series.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.rb_predictions = ctk.CTkRadioButton(self.options_graph_frame, text="Previsões", variable=self.graph_type, value="Previsões", command=self.update_options)
        self.rb_predictions.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.rb_losses = ctk.CTkRadioButton(self.options_graph_frame, text="Curvas de Aprendizado", variable=self.graph_type, value="Curvas de Aprendizado", command=self.update_options)
        self.rb_losses.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.rb_res = ctk.CTkRadioButton(self.options_graph_frame, text="Resíduos", variable=self.graph_type,
                                         value="Resíduos", command=self.update_options)
        self.rb_res.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.rb_simul = ctk.CTkRadioButton(self.options_graph_frame, text="Simulações", variable=self.graph_type,
                                            value="Simulações", command=self.update_options)
        self.rb_simul.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.options_frame = ctk.CTkFrame(self, fg_color="#EBEBEB")
        self.options_frame.pack(fill="both", expand=True, pady=(8,0), padx=5)

        self.series_checks = []
        self.prediction_checks = []
        self.simulation_checks = []
        self.losses_checks = []
        self.residual_checks = []
        self.residual_var = ctk.IntVar(value=-1)

        self.update_options()

        btn_frame = ctk.CTkFrame(self, fg_color='transparent')
        btn_frame.pack(pady=20)
        btn_cancel = ctk.CTkButton(btn_frame, text="Cancelar", command=self.destroy)
        btn_cancel.grid(row=0, column=0, padx=10)
        btn_confirm = ctk.CTkButton(btn_frame, text="Criar Gráfico", command=self.confirm)
        btn_confirm.grid(row=0, column=1, padx=10)

    def update_options(self):
        for widget in self.options_frame.winfo_children():
            widget.destroy()

        self.series_checks = []
        self.prediction_checks = []
        self.simulation_checks = []
        self.losses_checks = []
        self.residual_checks = []

        if self.graph_type.get() == "Séries":
            label_series = ctk.CTkLabel(self.options_frame, text="Selecione Séries:", font=ctk.CTkFont(size=13, weight="bold"))
            label_series.pack(anchor="w", pady=(10, 0), padx=10)

            for name in self.series.keys():
                var = ctk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.options_frame, text=name, variable=var)
                cb.pack(anchor="w", padx=20)
                self.series_checks.append(var)

        elif self.graph_type.get() == "Previsões":
            label_preds = ctk.CTkLabel(self.options_frame, text="Selecione Previsões:", font=ctk.CTkFont(size=13, weight="bold"))
            label_preds.pack(anchor="w", pady=(10, 0), padx=10)

            for name in self.predictions.keys():
                var = ctk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.options_frame, text=name, variable=var)
                cb.pack(anchor="w", padx=20)
                self.prediction_checks.append(var)

        elif self.graph_type.get() == "Curvas de Aprendizado":
            label_losses = ctk.CTkLabel(self.options_frame, text="Selecione Curvas de Aprendizado:", font=ctk.CTkFont(size=13, weight="bold"))
            label_losses.pack(anchor="w", pady=(10, 0), padx=10)

            for name in self.losses.keys():
                var = ctk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.options_frame, text=name, variable=var)
                cb.pack(anchor="w", padx=20)
                self.losses_checks.append(var)

        elif self.graph_type.get() == "Simulações":
            label_simul = ctk.CTkLabel(self.options_frame, text="Selecione Simulações:", font=ctk.CTkFont(size=13, weight="bold"))
            label_simul.pack(anchor="w", pady=(10, 0), padx=10)

            for name in self.simulations.keys():
                var = ctk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.options_frame, text=name, variable=var)
                cb.pack(anchor="w", padx=20)
                self.simulation_checks.append(var)

        else:  # Resíduos
            label_res = ctk.CTkLabel(self.options_frame, text="Selecione o resíduo (DARTS):", font=ctk.CTkFont(size=13, weight="bold"))
            label_res.pack(anchor="w", pady=(15, 5), padx=10)

            for i, (name, _) in enumerate(self.residuals.items()):
                rb = ctk.CTkRadioButton(self.options_frame, text=name, variable=self.residual_var, value=i)
                rb.pack(anchor="w", padx=20)
                self.residual_checks.append(rb)

    def confirm(self):
        name = self.name_entry.get().strip()
        if not name:
            return

        if self.graph_type.get() == "Séries":
            selected_names = [name for i, name in enumerate(self.series.keys()) if self.series_checks[i].get()]
            if selected_names:
                selected_series = {n: self.series[n] for n in selected_names}
                selected_data = {
                    "Séries": selected_series,
                    "Previsões": {},
                    "Simulações": {},
                    "Curvas de Aprendizado": {},
                    "Resíduos": None
                }
                self.callback(name, selected_data)
                self.destroy()

        elif self.graph_type.get() == "Previsões":
            selected_names = [name for i, name in enumerate(self.predictions.keys()) if
                              self.prediction_checks[i].get()]
            if selected_names:
                selected_predictions = {n: self.predictions[n] for n in selected_names}
                selected_data = {
                    "Séries": {},
                    "Previsões": selected_predictions,
                    "Simulações": {},
                    "Curvas de Aprendizado": {},
                    "Resíduos": None
                }
                self.callback(name, selected_data)
                self.destroy()

        elif self.graph_type.get() == "Curvas de Aprendizado":
            selected_names = [name for i, name in enumerate(self.losses.keys()) if self.losses_checks[i].get()]
            if selected_names:
                selected_losses = {n: self.losses[n] for n in selected_names}
                selected_data = {
                    "Séries": {},
                    "Previsões": {},
                    "Simulações": {},
                    "Curvas de Aprendizado": selected_losses,
                    "Resíduos": None
                }
                self.callback(name, selected_data)
                self.destroy()

        elif self.graph_type.get() == "Simulações":
            selected_names = [name for i, name in enumerate(self.simulations.keys()) if
                              self.simulation_checks[i].get()]
            if selected_names:
                selected_simulations = {n: self.simulations[n] for n in selected_names}
                selected_data = {
                    "Séries": {},
                    "Previsões": {},
                    "Simulações": selected_simulations,
                    "Curvas de Aprendizado": {},
                    "Resíduos": None
                }
                self.callback(name, selected_data)
                self.destroy()

        else:  # Resíduos
            idx = self.residual_var.get()
            if idx != -1:
                res_key = list(self.residuals.keys())[idx]
                selected_data = {
                    "Séries": {},
                    "Previsões": {},
                    "Simulações": {},
                    "Curvas de Aprendizado": {},
                    "Resíduos": {res_key: self.residuals[res_key]}
                }
                self.callback(name, selected_data)
                self.destroy()

    def centralize_window(self, width=330, height=560):
        window_width = width
        window_height = height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

class DeleteGraphsModal(ctk.CTkToplevel):
    def __init__(self, parent, graph_names, callback):
        super().__init__(parent)
        self.title("Excluir Gráficos")
        self.grab_set()
        self.centralize_window()

        self.callback = callback
        ctk.CTkLabel(self, text="Selecione os gráficos para excluir", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        self.check_vars = []
        self.graph_names = graph_names
        self.check_frame = ctk.CTkScrollableFrame(self)
        self.check_frame.pack(fill="both", expand=True, padx=10, pady=10)
        for name in graph_names:
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self.check_frame, text=name, variable=var)
            cb.pack(anchor="w", pady=2)
            self.check_vars.append((var, name))
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=15)
        btn_cancel = ctk.CTkButton(btn_frame, text="Cancelar", command=self.destroy)
        btn_cancel.grid(row=0, column=0, padx=10)
        btn_delete = ctk.CTkButton(btn_frame, text="Excluir Selecionados", command=self.delete_selected)
        btn_delete.grid(row=0, column=1, padx=10)

    def delete_selected(self):
        to_delete = [name for (var, name) in self.check_vars if var.get()]
        if to_delete:
            self.callback(to_delete)
            self.destroy()

    def centralize_window(self, width=320, height=400):
        window_width = width
        window_height = height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")
