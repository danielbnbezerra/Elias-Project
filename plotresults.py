import os
import tkinter as tk
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import save as torch_save
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from darts import TimeSeries
from series import MetricModels
import math

class PlotWindow(ctk.CTkToplevel):
    def __init__(self, series, predictions, residuals, losses):
        super().__init__()
        self.title("Visualização de Resultados")
        self.series = series
        self.predictions = predictions
        self.residuals = residuals
        self.losses = losses

        # Menu lateral
        self.menu_frame = ctk.CTkFrame(self, width=200)
        self.menu_frame.pack(side="left", fill="y")

        new_graph_button = ctk.CTkButton(self.menu_frame, text="Novo Gráfico", command=self.open_new_graph_modal)
        new_graph_button.pack(pady=(20,10), padx=20)

        delete_graph_button = ctk.CTkButton(self.menu_frame, text="Excluir Gráficos",
                                         command=self.open_delete_graphs_modal)
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
        report_menu.add_command(label="Gerar Relatório", command=self.export_report)
        report_menu.add_command(label="Exportar Modelos", command=self.export_models)

    def show_initial_graph_area(self):
        if hasattr(self, 'graph_area_frame'):
            self.graph_area_frame.destroy()
        self.graph_area_frame = ctk.CTkFrame(self)
        self.graph_area_frame.pack(side="right", fill="both", expand=True)
        self.initial_label = ctk.CTkLabel(
            self.graph_area_frame,
            text="Clique em 'Novo Gráfico' para adicionar gráficos",
            font=ctk.CTkFont(size=18)
        )
        self.initial_label.place(relx=0.5, rely=0.5, anchor="center")

    def open_new_graph_modal(self):
        CreateGraphModal(self, self.series, self.predictions, self.residuals, self.add_graph)

    def open_delete_graphs_modal(self):
        if not self.graphs:
            return
        names = [g["name"] for g in self.graphs]
        DeleteGraphsModal(self, names, self.delete_graphs)


    def add_graph(self, name, selected_data_indices):
        if hasattr(self, "initial_label") and self.initial_label.winfo_exists():
            self.graph_area_frame.destroy()
            self.graph_area_frame = ctk.CTkFrame(self)
            self.graph_area_frame.pack(side="right", fill="both", expand=True)
            self.graphs = []

        frame = ctk.CTkFrame(self.graph_area_frame, corner_radius=10, fg_color="#222222")

        if selected_data_indices.get("Resíduo") is not None and selected_data_indices["Resíduo"] != -1:
            idx = selected_data_indices["Resíduo"]
            res_obj = self.residuals_list[idx]  # Aqui, res_obj deve ser o residuals do DARTS

            # Gera o gráfico da análise de resíduos usando a função do DARTS
            plt.close('all')  # Fecha figuras abertas anteriormente para evitar acúmulo
            plot_residuals_analysis(res_obj["data"])
            fig = plt.gcf()
            fig.set_size_inches(4, 3)  # Ajusta para 4x3 polegadas como nos gráficos das séries
            # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margens para melhor preencher
            fig.suptitle(name, fontsize=14)  # título geral da figura
            # plot_residuals_analysis já cria a figura matplotlib e retorna ela

        else:
            fig = Figure(figsize=(4, 3), dpi=100)
            ax = fig.add_subplot(111)
            for idx in selected_data_indices["Séries"]:
                ax.plot(self.series_list[idx], label=f"Série {idx + 1}")
            for idx in selected_data_indices["Previsões"]:
                ax.plot(self.predictions_list[idx], label=f"Previsão {idx + 1}", linestyle='--')
            ax.legend()
            ax.grid(True)
            ax.set_title(name)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        self.graphs.append({"frame": frame, "canvas": canvas, "name": name})
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
        total = len(self.graphs)
        if total == 0:
            return
        cols = math.ceil(math.sqrt(total))
        rows = math.ceil(total / cols)

        for r in range(rows):
            self.graph_area_frame.grid_rowconfigure(r, weight=1)
        for c in range(cols):
            self.graph_area_frame.grid_columnconfigure(c, weight=1)

        for graph in self.graphs:
            graph["frame"].grid_forget()

        for idx, graph in enumerate(self.graphs):
            r = idx // cols
            c = idx % cols
            graph["frame"].grid(row=r, column=c, sticky="nsew", padx=5, pady=5)

    def centralize_window(self, width=1200,height=700):
        window_width = width
        window_height = height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

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

class CreateGraphModal(ctk.CTkToplevel):
    def __init__(self, parent, series_list, predictions_list, residuals_list, callback):
        super().__init__(parent)
        self.title("Novo Gráfico")
        self.geometry("450x560")
        self.callback = callback

        self.series_list = series_list
        self.predictions_list = predictions_list
        self.residuals_list = residuals_list  # Cada item: {"name": ..., "data": np.array}

        ctk.CTkLabel(self, text="Nome do gráfico:", font=ctk.CTkFont(size=14)).pack(pady=(10,0))
        self.name_entry = ctk.CTkEntry(self)
        self.name_entry.pack(pady=(0,15), padx=20, fill="x")
        self.name_entry.focus()

        ctk.CTkLabel(self, text="Tipo de gráfico:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(2,5))
        self.graph_type = ctk.StringVar(value="Série/Previsão")
        self.rb_sp = ctk.CTkRadioButton(self, text="Série/Previsão", variable=self.graph_type, value="Série/Previsão", command=self.update_options)
        self.rb_sp.pack(anchor="w", padx=30)
        self.rb_res = ctk.CTkRadioButton(self, text="Resíduo (DARTS)", variable=self.graph_type, value="Resíduo", command=self.update_options)
        self.rb_res.pack(anchor="w", padx=30)

        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(fill="both", expand=True, pady=(8,0), padx=5)

        self.sp_checks = {"Séries": [], "Previsões": []}
        self.res_check = []
        self.res_var = ctk.IntVar(value=-1)

        self.update_options()

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=20)
        btn_cancel = ctk.CTkButton(btn_frame, text="Cancelar", command=self.destroy)
        btn_cancel.grid(row=0, column=0, padx=10)
        btn_confirm = ctk.CTkButton(btn_frame, text="Criar Gráfico", command=self.confirm)
        btn_confirm.grid(row=0, column=1, padx=10)

    def update_options(self):
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        self.sp_checks = {"Séries": [], "Previsões": []}
        self.res_check = []

        if self.graph_type.get() == "Série/Previsão":
            lbl_series = ctk.CTkLabel(self.options_frame, text="Selecione Séries:", font=ctk.CTkFont(size=13, weight="bold"))
            lbl_series.pack(anchor="w", pady=(10, 0), padx=10)
            for i in range(len(self.series_list)):
                var = ctk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.options_frame, text=f"Série {i+1}", variable=var)
                cb.pack(anchor="w", padx=20)
                self.sp_checks["Séries"].append(var)

            lbl_preds = ctk.CTkLabel(self.options_frame, text="Selecione Previsões:", font=ctk.CTkFont(size=13, weight="bold"))
            lbl_preds.pack(anchor="w", pady=(15, 0), padx=10)
            for i in range(len(self.predictions_list)):
                var = ctk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.options_frame, text=f"Previsão {i+1}", variable=var)
                cb.pack(anchor="w", padx=20)
                self.sp_checks["Previsões"].append(var)
        else:
            lbl_res = ctk.CTkLabel(self.options_frame, text="Selecione o resíduo (DARTS):", font=ctk.CTkFont(size=13, weight="bold"))
            lbl_res.pack(anchor="w", pady=(15, 5), padx=10)
            for i, res in enumerate(self.residuals_list):
                rb = ctk.CTkRadioButton(self.options_frame, text=res["name"], variable=self.res_var, value=i)
                rb.pack(anchor="w", padx=20)
                self.res_check.append(rb)

    def confirm(self):
        name = self.name_entry.get().strip()
        if not name:
            return
        if self.graph_type.get() == "Série/Previsão":
            selected_data = {
                "Séries": [i for i, v in enumerate(self.sp_checks["Séries"]) if v.get()],
                "Previsões": [i for i, v in enumerate(self.sp_checks["Previsões"]) if v.get()],
                "Resíduo": None
            }
            if selected_data["Séries"] or selected_data["Previsões"]:
                self.callback(name, selected_data)
                self.destroy()
        else:
            idx = self.res_var.get()
            if idx != -1:
                selected_data = {
                    "Séries": [],
                    "Previsões": [],
                    "Resíduo": idx
                }
                self.callback(name, selected_data)
                self.destroy()

    def centralize_window(self, width=480, height=380):
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
        self.geometry("300x400")
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

    def centralize_window(self, width=480, height=380):
        window_width = width
        window_height = height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")
