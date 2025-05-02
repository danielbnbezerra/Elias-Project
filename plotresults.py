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

        # Layout com 2 colunas
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Submenu de seleção
        self.plot_menu_frame = ctk.CTkFrame(self)
        self.plot_menu_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        self.check_vars = {}
        self.make_checkboxes()

        # Área de plot
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.canvas = None

        self.centralize_window()
        self.bring_fwd_window()

    def make_checkboxes(self):
        models = list(self.predictions.keys())
        options = []
        for model in models:
            if model == "LHC":
                options.append(("LHC - Previsão", "lhc_pred")),
                options.append(("LHC - Curva de Aprendizado", "lhc_loss"))
            if model == "NBEATS":
                options.append(("N-BEATS - Previsão", "nbeats_pred")),
                options.append(("N-BEATS - Curva de Aprendizado", "nbeats_loss"))
            if model == "NHiTS":
                options.append(("N-HiTS - Previsão", "nhits_pred")),
                options.append(("N-HiTS - Curva de Aprendizado", "nhits_loss"))

        for text, key in options:
            self.check_vars[key] = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self.plot_menu_frame, text=text, variable=self.check_vars[key],
                                 command=self.update_plot)
            cb.pack(anchor="w", pady=4)

    def update_plot(self):
        selections = {k: v.get() for k, v in self.check_vars.items()}
        plots = []

        # Converter TimeSeries para numpy arrays e fazer a comparação
        selections = {k: v.get() for k, v in self.check_vars.items()}
        plots = []

        if selections.get("lhc_pred", False):
            plots.append((f"LHC\nMAPE: {self.metrics.mape['LHC']:.3f}%, RMSE: {self.metrics.rmse['LHC']:.3f}",
                          self.timeseries, self.predictions["LHC"], "blue"))

        if selections.get("nbeats_pred", False):
            plots.append((f"N-BEATS\nMAPE: {self.metrics.mape['NBEATS']:.3f}%, RMSE: {self.metrics.rmse['NBEATS']:.3f}",
                          self.timeseries, self.predictions["NBEATS"], "green"))

        if selections.get("nhits_pred", False):
            plots.append((f"N-HiTS\nMAPE: {self.metrics.mape['NHiTS']:.3f}%, RMSE: {self.metrics.rmse['NHiTS']:.3f}",
                          self.timeseries, self.predictions["NHiTS"], "red"))

        learning_curves = []
        if selections.get("lhc_loss", False):
            learning_curves.append(("LHC Loss", self.loss_curves["LHC"], "blue"))
        if selections.get("nbeats_loss", False):
            learning_curves.append(("N-BEATS Loss", self.loss_curves["NBEATS"], "green"))
        if selections.get("nhits_loss", False):
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