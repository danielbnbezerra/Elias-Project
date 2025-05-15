import os
import tkinter as tk
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import save as torch_save
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from darts import TimeSeries

from series import MetricModels

#FALTA INTEGRAR EXEMPLO 20 AQUI

class PlotWindow(ctk.CTkToplevel):
    def __init__(self, series, preds, losses):
        super(PlotWindow, self).__init__()
        self.grab_set()
        self.grid_propagate(True)
        self.create_submenu()
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
        self.selections = None

        # Área de plot
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        #Iniciando o canvas
        self.canvas = None

        self.centralize_window()
        self.bring_fwd_window()

    def create_submenu(self):
        #Export Menu
        export_menu = tk.Menu(self)
        self.config(menu=export_menu)

        #Report Menu
        report_menu = tk.Menu(export_menu, tearoff=0)
        export_menu.add_cascade(label="Resultados", menu=report_menu)
        report_menu.add_command(label="Gerar Relatório", command=self.export_report)
        report_menu.add_command(label="Exportar Modelos", command=self.export_models)

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

    def update_selection(self):
        self.selections = {k: v.get() for k, v in self.check_vars.items()}

    def update_plot(self):
        plots = []
        self.update_selection()

        if self.selections.get("lhc_pred", False):
            plots.append((f"LHC\nMAPE: {self.metrics.mape['LHC']:.3f}%, RMSE: {self.metrics.rmse['LHC']:.3f}",
                          self.timeseries, self.predictions["LHC"], "blue"))

        if self.selections.get("nbeats_pred", False):
            plots.append((f"N-BEATS\nMAPE: {self.metrics.mape['NBEATS']:.3f}%, RMSE: {self.metrics.rmse['NBEATS']:.3f}",
                          self.timeseries, self.predictions["NBEATS"], "green"))

        if self.selections.get("nhits_pred", False):
            plots.append((f"N-HiTS\nMAPE: {self.metrics.mape['NHiTS']:.3f}%, RMSE: {self.metrics.rmse['NHiTS']:.3f}",
                          self.timeseries, self.predictions["NHiTS"], "red"))

        learning_curves = []
        if self.selections.get("lhc_loss", False):
            learning_curves.append(("LHC Loss", self.loss_curves["LHC"], "blue"))
        if self.selections.get("nbeats_loss", False):
            learning_curves.append(("N-BEATS Loss", self.loss_curves["NBEATS"], "green"))
        if self.selections.get("nhits_loss", False):
            learning_curves.append(("N-HiTS Loss", self.loss_curves["NHiTS"], "red"))

        total_plots = len(plots) + (1 if learning_curves else 0)

        if total_plots == 0:
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            return

        if total_plots == 1:
            rows, cols = 1, 1
        else:
            cols = 2
            rows = (total_plots + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))

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

    def export_models(self):
        for name, model in self.models.items():
            if name == "LHC":
                torch_save(model.state_dict(), "lhc_model_weights.pth")
                with open("lhc_model_class.py", "w") as f:
                    f.write(inspect.getsource(type(model)))
            elif name in ("NBEATS", "NHiTS"):
                path = name.lower()
                py_code = f"""
    from darts.models import {type(model).__name__}

    def load_model():
        model = {type(model).__name__}(
            input_chunk_length={model.input_chunk_length},
            output_chunk_length={model.output_chunk_length},
            n_epochs={model.n_epochs},
            random_state=42
        )
        model.load_model('{path}_weights.pth.tar')
        return model
    """
                with open(f"{path}_model.py", "w") as f:
                    f.write(py_code)
                    model.save_model(f"{path}_weights.pth.tar")

    def export_report(self):

        # Geração de nome único para o PDF
        pdf_filename = self.get_unique_filename("Relatório", "pdf")

        # PDF: salvar previsões individualmente e as curvas de aprendizado juntas
        with PdfPages(pdf_filename) as pdf:
            plot_configs = []

            # Gráficos de previsão
            if self.selections.get("lhc_pred", False):
                plot_configs.append(("LHC - Previsão", self.predictions.get("LHC"), "blue"))
            if self.selections.get("nbeats_pred", False):
                plot_configs.append(("N-BEATS - Previsão", self.predictions.get("NBEATS"), "green"))
            if self.selections.get("nhits_pred", False):
                plot_configs.append(("N-HiTS - Previsão", self.predictions.get("NHiTS"), "red"))

            for title, pred, color in plot_configs:
                if pred is None:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(self.timeseries.time_index, self.timeseries.values(), label="Original", color="black")
                ax.plot(pred.time_index, pred.values(), label="Previsão", color=color)
                ax.set_title(title)
                ax.legend()
                pdf.savefig(fig)
                plt.close(fig)

            # Gráfico único com todas as curvas de aprendizado
            learning_curves = []

            if self.selections.get("lhc_loss", False):
                learning_curves.append(("LHC Loss", self.loss_curves["LHC"], "blue"))
            if self.selections.get("nbeats_loss", False):
                learning_curves.append(("N-BEATS Loss", self.loss_curves["NBEATS"], "green"))
            if self.selections.get("nhits_loss", False):
                learning_curves.append(("N-HiTS Loss", self.loss_curves["NHiTS"], "red"))

            fig, ax = plt.subplots(figsize=(8, 4))
            for label, loss, color in learning_curves:
                if loss is not None:
                    ax.plot(range(len(loss)), loss, label=label, color=color)

            ax.set_title("Curvas de Aprendizado")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Perda")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

        # Métricas
        results_metrics = []
        for name in self.metrics.mape:
            print(name)
            results_metrics.append({"Modelo": name.upper(), "MAPE": self.metrics.mape[name], "RMSE": self.metrics.rmse[name]})

        df = pd.DataFrame(results_metrics)
        df.to_excel("Métricas.xlsx", index=False)

    def get_unique_filename(self,base_name, extension):
        filename = f"{base_name}.{extension}"
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_name} ({counter}).{extension}"
            counter += 1
        return filename

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