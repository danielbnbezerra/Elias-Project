import inspect
import math
import os
import tempfile

import tkinter as tk
import matplotlib.dates as mdates
import customtkinter as ctk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from darts.utils.statistics import plot_residuals_analysis, plot_acf, plot_pacf
from torch import save as torch_save

from metrics import *


class PlotWindow(ctk.CTkToplevel):
    def __init__(self, series, predictions, residuals, losses, models):
        super().__init__()
        self.title("Visualização de Resultados")
        self.grab_set()
        self.create_submenu()
        self.timeseries = series

        self.series = {"Precipitação":series.prate,
                       "Precipitação Acumulada 3 dias": series.prate_t_minus_3,
                       "Precipitação Acumulada 5 dias": series.prate_t_minus_5,
                       "Vazão":series.flow}

        self.predictions = predictions
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

    def create_submenu(self):
        #Export Menu
        export_menu = tk.Menu(self)
        self.config(menu=export_menu)

        #Report Menu
        report_menu = tk.Menu(export_menu, tearoff=0)
        export_menu.add_cascade(label="Resultados", menu=report_menu)
        report_menu.add_command(label="Gerar Relatório", command=self.generate_report_pdf)
        report_menu.add_command(label="Exportar Modelos", command=self.export_models)

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

            ax.set_xticks(
                pd.date_range(start=self.series["Vazão"].start_time(), end=self.series["Vazão"].end_time(), freq='MS'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.set_xlabel('Tempo', fontsize=15)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True)
            ax.set_title(name_entry)

        # Previsões (sempre plota flow para comparação)
        elif selected_data_indices.get("Previsões"):
            # Plota série flow inteira com tempo original para comparação
            ax.plot(self.series["Vazão"].time_index, self.series["Vazão"].values(), label="Série Observada")

            # Para cada previsão, usa seu time_index para alinhamento correto no tempo
            for name, pred_series in selected_data_indices["Previsões"].items():
                ax.plot(pred_series.time_index, pred_series.values(), label=f"{name} - Previsão")

            ax.set_xticks(
                pd.date_range(start=self.series["Vazão"].start_time(), end=self.series["Vazão"].end_time(), freq='YS'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.set_xlabel('Tempo', fontsize=15)
            ax.set_ylabel('Vazão', fontsize=15)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True)
            ax.set_title(name_entry)

        # Curvas de Aprendizado
        elif selected_data_indices.get("Curvas de Aprendizado"):
            for name, loss_values in selected_data_indices["Curvas de Aprendizado"].items():
                ax.plot(range(len(loss_values)), loss_values, label=f"{name} - Curva de Aprendizado")
            ax.set_xlabel("Epochs", fontsize=15)
            ax.set_ylabel("Perdas", fontsize=15)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True)
            ax.set_title(name_entry)

        # Resíduos (único selecionado)
        elif selected_data_indices.get("Resíduos"):
            for name, res_obj in selected_data_indices["Resíduos"].items():
                plt.close('all')
                plot_residuals_analysis(res_obj)
                fig = plt.gcf()
                fig.set_size_inches(8, 6)
                fig.suptitle(f"{name} - Resíduos", fontsize=14)
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

    def generate_report_pdf(self, name_file=f"Relatório Completo"):
        filename = self.get_unique_filename(name_file,"pdf")
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        margin = 50
        line_height = 14

        y_position = height - margin

        for model_name, predicted_ts in self.predictions.items():
            actual_ts = self.timeseries.valid_target

            actual = actual_ts.values().flatten()

            rmse_value = rmse(actual_ts, predicted_ts)
            mae_value = mae(actual_ts, predicted_ts)
            nse_value = nse(actual_ts, predicted_ts)
            kge_value = kge(actual_ts, predicted_ts)

            mk_result = mann_kendall_test(actual)

            dagostino_result = dagostino_k_squared_test(self.residuals[model_name])
            anderson_result = anderson_darling_test(self.residuals[model_name])
            shapiro_result = shapiro_wilk_test(self.residuals[model_name])

            adf_result = adf_test(actual)
            kpss_result = kpss_test(actual)

            # Gerar gráficos ACF e PACF e salvar imagens temporárias
            with tempfile.TemporaryDirectory() as tmpdir:
                acf_path = os.path.join(tmpdir, "acf.png")
                pacf_path = os.path.join(tmpdir, "pacf.png")

                plt.figure(figsize=(6, 3))
                plot_acf(actual_ts, max_lag=40)
                plt.suptitle("Autocorrelação", fontsize=14)
                plt.tight_layout()
                plt.savefig(acf_path)
                plt.close()

                plt.figure(figsize=(6, 3))
                plot_pacf(actual_ts, max_lag=40)
                plt.suptitle("Autocorrelação Parcial", fontsize=14)
                plt.tight_layout()
                plt.savefig(pacf_path)
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
                    f"Tendência (Mann-Kendall):",
                    f"",
                    f"{mk_result['trend']}, Tau: {mk_result['tau']:.4f}, p-value: {mk_result['p']:.4f}",
                    f"",
                    f"",
                    f"Normalidade dos Resíduos:",
                    f"",
                    f" - D’Agostino: {dagostino_result['result']} (p-value={dagostino_result.get('p_value', 'N/A'):.4f})",
                    f" - Anderson-Darling: {anderson_result['result']} (Estatística={anderson_result.get('statistic', 'N/A'):.4f}, Critério 5%={anderson_result.get('critical_value_5pct', 'N/A'):.4f})",
                    f" - Shapiro-Wilk: {shapiro_result['result']} (p-value={shapiro_result.get('p_value', 'N/A'):.4f})",
                    f"",
                    f"",
                    f"Estacionariedade:",
                    f"",
                    f" - ADF: {adf_result['result']} (Estatística={adf_result['adf_statistic']:.4f}, p-value={adf_result['p_value']:.4f})",
                    f" - KPSS: {kpss_result['result']} (Estatística={kpss_result['kpss_statistic']:.4f}, p-value={kpss_result['p_value']:.4f})",
                ]

                for line in lines:
                    if y_position < margin:
                        c.showPage()
                        y_position = height - margin
                        c.setFont("Helvetica", 12)
                    c.drawString(margin, y_position, line)
                    y_position -= line_height

                # Inserir imagens dos gráficos
                if y_position < 200:
                    c.showPage()
                    y_position = height - margin

                acf_img = ImageReader(acf_path)
                pacf_img = ImageReader(pacf_path)

                c.drawImage(acf_img, margin, y_position - 150, width=250, height=150)
                c.drawImage(pacf_img, margin + 270, y_position - 150, width=250, height=150)
                y_position -= 170

                # Linha separadora
                c.line(margin, y_position, width - margin, y_position)
                y_position -= line_height

        c.save()
        print(f"Relatório salvo em: {filename}")

    def get_unique_filename(self, base_name, extension):
        filename = f"{base_name}.{extension}"
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_name} ({counter}).{extension}"
            counter += 1
        return filename

class CreateGraphModal(ctk.CTkToplevel):
    def __init__(self, parent, series, predictions, losses, residuals, callback):
        super().__init__(parent)
        self.title("Novo Gráfico")
        self.grab_set()
        self.centralize_window()

        self.callback = callback

        self.series_names = list(series.keys())
        self.series = series
        self.predictions = predictions
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

        self.options_frame = ctk.CTkFrame(self, fg_color="#EBEBEB")
        self.options_frame.pack(fill="both", expand=True, pady=(8,0), padx=5)

        self.series_checks = []
        self.prediction_checks = []
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
                    "Curvas de Aprendizado": selected_losses,
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
