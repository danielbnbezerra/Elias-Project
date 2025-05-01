import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from darts import TimeSeries

from series import MetricModels

#FALTA INTEGRAR EXEMPLO 20 AQUI

class PlotWindow(ctk.CTkToplevel):
    def __init__(self, series, preds, losses):
        super(PlotWindow, self).__init__()
        self.grab_set()
        self.grid_propagate(True)
        self.timeseries = series.timeseries
        self.predictions = preds
        self.loss_curves = losses
        self.metrics = MetricModels(series.valid, preds)
        self.title("Visualização de Resultados")
        self.geometry("1000x600")
        print("COMEÇO")
        print("série")
        print(self.timeseries)
        print("previsão")
        print(self.predictions)
        print("perdas")
        print(self.loss_curves)
        print("métricas")
        print(self.metrics.mape,self.metrics.rmse)

        # Layout com 2 colunas
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Submenu de seleção
        self.plot_menu_frame = ctk.CTkFrame(self)
        self.plot_menu_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        self.check_vars = {}
        options = [
            ("LSTM - Previsão", "lstm_pred"),
            ("N-BEATS - Previsão", "nbeats_pred"),
            ("N-HiTS - Previsão", "nhits_pred"),
            ("LSTM - Curva de Aprendizado", "lstm_loss"),
            ("N-BEATS - Curva de Aprendizado", "nbeats_loss"),
            ("N-HiTS - Curva de Aprendizado", "nhits_loss"),
        ]

        for text, key in options:
            self.check_vars[key] = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self.plot_menu_frame, text=text, variable=self.check_vars[key], command=self.update_plot)
            cb.pack(anchor="w", pady=4)

        # Área de plot
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.canvas = None

    def update_plot(self):
        selections = {k: v.get() for k, v in self.check_vars.items()}
        plots = []

        # def get_metrics(y_true, y_pred):
        #     mae = mean_absolute_error(y_true, y_pred)
        #     rmse = mean_squared_error(y_true, y_pred) ** 0.5
        #     return mae, rmse

        # Converter TimeSeries para numpy arrays e fazer a comparação
        if selections["lstm_pred"]:
            plots.append((f"LSTM\nMAPE: {self.metrics.mape['LHC']:.3f}%, RMSE: {self.metrics.rmse['LHC']:.3f}", self.timeseries, self.predictions["LHC"], "blue"))

        if selections["nbeats_pred"]:
            plots.append((f"N-BEATS\nMAPE: {self.metrics.mape['NBEATS']:.3f}%, RMSE: {self.metrics.rmse['NBEATS']:.3f}", self.timeseries, self.predictions["NBEATS"], "green"))

        if selections["nhits_pred"]:
            plots.append((f"N-HiTS\nMAPE: {self.metrics.mape['NBEATS']:.3f}%, RMSE: {self.metrics.rmse['NHiTS']:.3f}", self.timeseries, self.predictions["NHiTS"], "red"))

        learning_curves = []
        if selections["lstm_loss"]:
            learning_curves.append(("LSTM Loss", self.loss_curves["LHC"], "blue"))
        if selections["nbeats_loss"]:
            learning_curves.append(("N-BEATS Loss", self.loss_curves["NBEATS"], "green"))
        if selections["nhits_loss"]:
            learning_curves.append(("N-HiTS Loss", self.loss_curves["NHiTS"], "red"))

        total_plots = len(plots) + (1 if learning_curves else 0)

        if total_plots == 0:
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            return

        # Calcular rows e cols de forma adequada
        if total_plots == 1:
            rows, cols = 1, 1
        else:
            cols = 2
            rows = (total_plots + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))

        # Garantir que axs seja sempre uma lista
        if isinstance(axs, plt.Axes):
            axs = [axs]
        else:
            axs = axs.flatten()

        i = 0
        for title, base, pred, color in plots:
            axs[i].plot(base.time_index, base.values(), label="Original", color='black')
            axs[i].plot(pred.time_index, pred.values(), label="Previsão", color=color)
            axs[i].set_title(title)
            axs[i].legend()
            i += 1

        if learning_curves:
            for name, loss, color in learning_curves:
                axs[i].plot(range(len(loss)), loss, label=name, color=color)
            axs[i].set_title("Curvas de Aprendizado")
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel("Perda")
            axs[i].legend()
            i += 1

        for j in range(i, len(axs)):
            fig.delaxes(axs[j])

        fig.tight_layout()

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def centralize_window(self):
        # window_width = round(self.winfo_width(),-1)
        # window_height = round(self.winfo_height(),-1)
        window_width = 1000
        window_height = 600
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

    def bring_fwd_window(self):
        self.attributes("-topmost", True)