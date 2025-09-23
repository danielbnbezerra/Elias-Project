import shutil
import ast

from tkinter import messagebox, filedialog

from models import *


class Tooltip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.delay = delay
        self.after_id = None

        # Bind mouse events to the widget
        self.widget.bind("<Enter>", self.schedule_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<Motion>", self.move_tooltip)

    def schedule_tooltip(self, event=None):
        # Schedule the tooltip to appear after a delay
        self.after_id = self.widget.after(self.delay, self.show_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return  # Tooltip is already visible

        # Create a new Toplevel window for the tooltip
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations
        self.tooltip_window.attributes("-topmost", True)  # Keep on top

        # Position the tooltip near the widget
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tooltip_window.geometry(f"+{x}+{y}")

        # Add the tooltip text
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="lightyellow",
            relief="solid",
            borderwidth=1,
            font=("Arial", 10)
        )
        label.pack()

    def hide_tooltip(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)  # Cancel the scheduled tooltip
            self.after_id = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def move_tooltip(self, event=None):
        if self.tooltip_window:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + 20
            self.tooltip_window.geometry(f"+{x}+{y}")

class ConfirmExitWindow(ctk.CTkToplevel):
    @staticmethod
    def cleanup_darts_logs():

        folders_to_delete = ["darts_logs", "checkpoints"]
        for folder in folders_to_delete:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    print(f"Erro ao remover pasta {folder}: {e}")

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Sair do Aplicativo?")
        self.centralize_window()
        self.grab_set()  # Make this window modal (prevents interaction with the main window)
        self.parent = parent

        label = ctk.CTkLabel(self, text="Tem certeza que deseja sair?")
        label.pack(pady=20)

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=10)

        yes_button = ctk.CTkButton(button_frame, text="Sim", command=self.on_closing)
        yes_button.grid(row=0, column=0, padx=10)

        no_button = ctk.CTkButton(button_frame, text="Não", command=self.destroy)
        no_button.grid(row=0, column=1, padx=10)

    def on_closing(self):
        self.cleanup_darts_logs()
        self.parent.quit()  # fecha a janela

    def centralize_window(self, width=320, height=150):
        window_width = width
        window_height = height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

class BasicWindow(ctk.CTkToplevel):
    def __init__(self, series, index, remaining_models, configs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grab_set()
        self.grid_propagate(True)
        self.index = index
        self.selected_models = remaining_models
        self.model = None
        self.timeseries = series
        self.parameters = None
        self.configurations = configs if configs is not None else []

        self.option_frame = ctk.CTkFrame(self, fg_color="#DDDDDD")
        self.option_frame.place(x=0, y=0, relwidth=0.2, relheight=1)
        self.option_frame.columnconfigure(0, weight=1)
        self.hiperparameter_frame = ctk.CTkFrame(self, fg_color="#EBEBEB")
        self.hiperparameter_frame.place(relx=0.2, y=0, relwidth=0.8, relheight=1)


        #Configuração
        self.label_confg_buttons= ctk.CTkLabel(master=self.option_frame,
                                               text="Configuração:",
                                               font= ("Arial", 14))
        self.label_confg_buttons.grid(row=1, column=0, padx=5, pady=5)
        self.confgs_options = ["Opção 1", "Opção 2", "Opção 3", "Manual"]
        self.confgs_options = ctk.CTkOptionMenu(master=self.option_frame, values=self.confgs_options, command=self.confg_event)
        self.confgs_options.set("Selecione")
        self.confgs_options.grid(row=2, column=0, padx=5, pady=5)

        #Limpar
        self.clear_button = ctk.CTkButton(master=self.option_frame, text="Limpar", command=self.clean_parameters)
        self.clear_button.configure(state="disabled")
        self.clear_button.grid(row=3, column=0, pady=10)

        # Confirmar ou Próximo
        if self.index > (len(self.selected_models) - 1):
            self.confirm_button = ctk.CTkButton(master=self.option_frame, text="Confirmar", command=self.model_run)
            self.confirm_button.grid(row=4, column=0, pady=10)
            Tooltip(self.confirm_button, text="Confirmar escolhas e executar o modelo.")
        else:
            self.next_button = ctk.CTkButton(master=self.option_frame, text="Próximo", command=self.next_window)
            self.next_button.grid(row=4, column=0, pady=10)
            Tooltip(self.next_button, text="Passa para a próxima janela de modelo a ser configurada.")

        Tooltip(self.label_confg_buttons, text="Selecione uma configuração predeterminada\n "
                                               "para o modelo ou edite manualmente os \n"
                                               "parâmetros:")
        Tooltip(self.clear_button, text="Limpar todos as escolhas de hiperparâmetros.")

    def get_configurations(self):
        if self.get_parameters():
            for name, value in self.parameters.items():
                print(name, value, type(value))
            self.configurations.append(
                {"model": self.selected_models[self.index - 1]["name"], "parameters": self.parameters})
            return True
        else:
            return False

    def next_window(self):
        # Fecha a janela atual e abre a próxima
        if self.get_configurations():
            next_model_window = self.selected_models[self.index]["window"]
            # Cria a nova janela ANTES de destruir a atual
            next_model_window(self.timeseries, self.index + 1, self.selected_models, self.configurations)
            # Aguarda um pouco para garantir que a nova janela foi criada completamente
            self.after(100, self.destroy)
        else:
            # Se houver erro na obtenção de configurações, não fecha a janela
            return

    def centralize_window(self, width=1060,height=200):
        window_width = width
        window_height = height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width)//2,-1)
        y = round((screen_height - window_height)//2,-1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

    def model_run(self):
        if self.get_configurations():
            inicial_model_run = self.configurations[0]
            self.configurations.pop(0)
            if inicial_model_run["model"] == "LHC":
                ModelRunLHCWindow(inicial_model_run["parameters"], self.configurations, self.timeseries)
            if inicial_model_run["model"] == "N-BEATS":
                ModelRunNBEATSWindow(inicial_model_run["parameters"], self.configurations, self.timeseries)
            if inicial_model_run["model"] == "N-HiTS":
                ModelRunNHiTSWindow(inicial_model_run["parameters"], self.configurations, self.timeseries)
            self.after(100, self.destroy)
        else:
            return False

class LHCModelWindow(BasicWindow):
    def __init__(self, series, index, remaining_models, *args, **kwargs):
        super().__init__(series, index, remaining_models,*args, **kwargs)
        self.title("LHC - Escolha os Parâmetros")
        self.centralize_window(760, 200)
        self.option_frame.place(x=0, y=0, relwidth=0.25, relheight=1)
        self.hiperparameter_frame.place(relx=0.25, y=0, relwidth=0.75, relheight=1)

        # Random State
        self.random_state = 42

        # Input Length
        self.label_input_length = ctk.CTkLabel(master=self.hiperparameter_frame, text="Input Length:",
                                                      font=("Arial", 14))
        self.label_input_length.grid(row=0, column=0, padx=5, pady=5)
        self.entry_input_length = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11),
                                                      state="disabled")
        self.entry_input_length.grid(row=0, column=1, padx=5, pady=5)

        # Output Length
        self.label_output_length = ctk.CTkLabel(master=self.hiperparameter_frame, text="Output Length:", font=("Arial", 14))
        self.label_output_length.grid(row=1, column=0, padx=5, pady=5)
        self.entry_output_length = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_output_length.grid(row=1, column=1, padx=5, pady=5)

        # Hidden Size
        self.label_hidden_size = ctk.CTkLabel(master=self.hiperparameter_frame, text="Hidden Size:",
                                             font=("Arial", 14))
        self.label_hidden_size.grid(row=2, column=0, padx=5, pady=5)
        self.entry_hidden_size = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_hidden_size.grid(row=2, column=1, padx=5, pady=5)

        # Number of Layers
        self.label_num_layers = ctk.CTkLabel(master=self.hiperparameter_frame, text="Number of Layers:",
                                             font=("Arial", 14))
        self.label_num_layers.grid(row=3, column=0, padx=5, pady=5)
        self.entry_num_layers = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_num_layers.grid(row=3, column=1, padx=5, pady=5)

        # Dropout
        self.label_dropout = ctk.CTkLabel(master=self.hiperparameter_frame, text="Dropout:", font=("Arial", 14))
        self.label_dropout.grid(row=4, column=0, padx=5, pady=5)
        self.entry_dropout = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_dropout.grid(row=4, column=1, padx=5, pady=5)

        # Learning Rate
        self.label_learning_rate = ctk.CTkLabel(master=self.hiperparameter_frame, text="Learning Rate:",
                                              font=("Arial", 14))
        self.label_learning_rate.grid(row=0, column=2, padx=5, pady=5)
        self.entry_learning_rate = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11),
                                            state="disabled")
        self.entry_learning_rate.grid(row=0, column=3, padx=5, pady=5)

        # Batch Size
        self.label_batch_size = ctk.CTkLabel(master=self.hiperparameter_frame, text="Batch Size:", font=("Arial", 14))
        self.label_batch_size.grid(row=1, column=2, padx=5, pady=5)
        self.batch_sizes = ['16', '32', '64', '128', '256', '512', '1024']
        self.option_batch_size = ctk.CTkOptionMenu(master=self.hiperparameter_frame, values=self.batch_sizes,
                                                   state="disabled")
        self.option_batch_size.set("Selecione")
        self.option_batch_size.grid(row=1, column=3, padx=5, pady=5)

        # Number of Epochs
        self.label_n_epochs = ctk.CTkLabel(master=self.hiperparameter_frame, text="Num Epochs:", font=("Arial", 14))
        self.label_n_epochs.grid(row=2, column=2, padx=5, pady=5)
        self.n_epochs = ['100', '200', '300', '400', '500', '1000']
        self.option_n_epochs = ctk.CTkComboBox(master=self.hiperparameter_frame, values=self.n_epochs, state="disabled")
        self.option_n_epochs.set("Selecione/Digite")
        self.option_n_epochs.grid(row=2, column=3, padx=5, pady=5)

        # Save Checkpoints
        self.label_save_checkpoints = ctk.CTkLabel(master=self.hiperparameter_frame, text="Save Checkpoints:",
                                                   font=("Arial", 14))
        self.label_save_checkpoints.grid(row=3, column=2, padx=5, pady=5)
        self.boolean_options = ['True', 'False']
        self.option_save_checkpoints = ctk.CTkOptionMenu(master=self.hiperparameter_frame, values=self.boolean_options,
                                                         state="disabled")
        self.option_save_checkpoints.set("Selecione")
        self.option_save_checkpoints.grid(row=3, column=3, padx=5, pady=5)

        #Tooltips
        Tooltip(self.label_input_length, text="Quantidade de passos anteriores que o modelo utiliza para fazer a previsão em cada amostra.\n"
                                              "Janelas maiores capturam mais histórico, mas podem aumentar o tempo de treino. Valores inteiros.\n"
                                              "Recomenda-se de 10 a 60.")
        Tooltip(self.label_output_length, text="Quantidade de passos futuros que o modelo vai aprender a prever durante\n"
                                               "o treinamento em cada amostra. Valores Inteiros. Recomenda-se de 1 a 10.")
        Tooltip(self.label_hidden_size,text="Quantos neurônios cada camada LSTM terá. Mais neurônios podem capturar padrões complexos, mas tornam o modelo mais pesado.\n"
                                            "Valores inteiros. Recomenda-se potências de base 2, de 32 (2\u2075) a 512 (2\u2079)")
        Tooltip(self.label_num_layers, text="Quantas camadas da rede LSTM serão empilhadas. Camadas extras aumentam a capacidade do modelo,\n"
                                            "mas podem causar dificuldades no treino. Valores inteiros. Recomenda-se de 1 a 4.")
        Tooltip(self.label_dropout, text="A probabilidade de dropout a ser utilizada nas camadas totalmente conectadas.\n"
                                         "Recomenda-se entre 0.0 e 0.5.")
        Tooltip(self.label_batch_size, text="Número de amostras processadas por vez até que a atualização\n"
                                            "dos pesos em cada passagem de treinamento seja realizada.")
        Tooltip(self.label_learning_rate, text="Define a taxa de aprendizado por época.")
        Tooltip(self.label_n_epochs, text="Número de épocas em cada rodada de treinamento do modelo.")
        Tooltip(self.label_save_checkpoints, text="Define se o modelo não treinado e os checkpoints do treinamento serão salvos automaticamente.")


    def disable_parameters(self):
        self.entry_input_length.configure(state="disabled")
        self.entry_output_length.configure(state="disabled")
        self.entry_hidden_size.configure(state="disabled")
        self.entry_num_layers.configure(state="disabled")
        self.entry_dropout.configure(state="disabled")
        self.option_batch_size.configure(state="disabled")
        self.entry_learning_rate.configure(state="disabled")
        self.option_n_epochs.configure(state="disabled")
        self.option_save_checkpoints.configure(state="disabled")

    def enable_parameters(self):
        self.entry_input_length.configure(state="normal")
        self.entry_output_length.configure(state="normal")
        self.entry_hidden_size.configure(state="normal")
        self.entry_num_layers.configure(state="normal")
        self.entry_dropout.configure(state="normal")
        self.option_batch_size.configure(state="normal")
        self.entry_learning_rate.configure(state="normal")
        self.option_n_epochs.configure(state="normal")
        self.option_save_checkpoints.configure(state="normal")

    def confg_event(self, choice):
        if choice == 'Opção 1':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_length.insert(0, "1")
            self.entry_output_length.insert(0, "7")
            self.entry_hidden_size.insert(0,"32")
            self.entry_num_layers.insert(0, "1")
            self.entry_dropout.insert(0, "0.0")
            self.entry_learning_rate.insert(0,"0.01")
            self.option_batch_size.set("32")
            self.option_n_epochs.set("100")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 2':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_length.insert(0, "1")
            self.entry_output_length.insert(0, "7")
            self.entry_hidden_size.insert(0, "32")
            self.entry_num_layers.insert(0, "1")
            self.entry_dropout.insert(0, "0.0")
            self.entry_learning_rate.insert(0, "0.01")
            self.option_batch_size.set("32")
            self.option_n_epochs.set("100")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 3':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_length.insert(0, "1")
            self.entry_output_length.insert(0, "7")
            self.entry_hidden_size.insert(0, "32")
            self.entry_num_layers.insert(0, "1")
            self.entry_dropout.insert(0, "0.0")
            self.entry_learning_rate.insert(0, "0.01")
            self.option_batch_size.set("32")
            self.option_n_epochs.set("100")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Manual':
            self.clear_button.configure(state="normal")
            self.enable_parameters()
            self.clean_parameters()

    def clean_parameters(self):
        self.entry_input_length.delete(0, "end")
        self.entry_output_length.delete(0, "end")
        self.entry_hidden_size.delete(0, "end")
        self.entry_num_layers.delete(0, "end")
        self.entry_dropout.delete(0, "end")
        self.entry_learning_rate.delete(0, "end")
        self.option_batch_size.set("Selecione")
        self.option_n_epochs.set("Selecione/Digite")
        self.option_save_checkpoints.set("Selecione")

    def get_parameters(self):
        # --- Pega os valores dos widgets ---
        input_len_str = self.entry_input_length.get()
        output_len_str = self.entry_output_length.get()
        hidden_size_str = self.entry_hidden_size.get()
        num_layers_str = self.entry_num_layers.get()
        dropout_str = self.entry_dropout.get()
        learning_rate_str = self.entry_learning_rate.get()
        batch_size_str = self.option_batch_size.get()
        n_epochs_str = self.option_n_epochs.get()
        save_checkpoints_str = self.option_save_checkpoints.get()

        # --- Estágio de Validação (Antes de converter) ---
        # Verifica se todos os campos de texto estão preenchidos
        all_fields = {
            "Input Length": input_len_str,
            "Output Length": output_len_str,
            "Hidden Size": hidden_size_str,
            "Number of Layers": num_layers_str,
            "Dropout": dropout_str,
            "Learning Rate": learning_rate_str,
        }
        for field_name, value in all_fields.items():
            if not value:
                messagebox.showerror("Erro de Validação", f"O campo '{field_name}' não pode estar vazio.")
                return False

        # Verifica os campos de seleção
        if batch_size_str == "Selecione":
            messagebox.showerror("Erro de Validação", "Por favor, selecione um 'Batch Size'.")
            return False
        if n_epochs_str == "Selecione/Digite":
            messagebox.showerror("Erro de Validação", "Por favor, selecione ou digite o 'Num Epochs'.")
            return False
        if save_checkpoints_str == "Selecione":
            messagebox.showerror("Erro de Validação", "Por favor, selecione uma opção em 'Save Checkpoints'.")
            return False

        # --- Estágio de Conversão e Lógica ---
        try:
            # Conversão segura para os tipos corretos
            input_len = int(input_len_str)
            output_len = int(output_len_str)
            hidden_size = int(hidden_size_str)
            num_layers = int(num_layers_str)
            dropout = float(dropout_str)
            learning_rate = float(learning_rate_str)
            batch_size = int(batch_size_str)
            n_epochs = int(n_epochs_str)
            save_checkpoints = save_checkpoints_str == 'True'  # Converte para booleano

            # Validações lógicas
            if input_len <= 0 or output_len <= 0 or hidden_size <= 0 or num_layers <= 0 or batch_size <= 0 or n_epochs <= 0:
                messagebox.showerror("Erro de Validação", "Valores de parâmetros numéricos devem ser maiores que zero.")
                return False

            if not (0.0 <= dropout < 1.0):
                messagebox.showerror("Erro de Validação", "O valor de 'Dropout' deve estar entre 0.0 e 0.99.")
                return False

            if learning_rate <= 0:
                messagebox.showerror("Erro de Validação", "O 'Learning Rate' deve ser maior que zero.")
                return False

            if (input_len + output_len) > len(self.timeseries.valid_target):
                messagebox.showerror("Erro de Validação",
                                     f"A soma de Input ({input_len}) e Output ({output_len}) não pode ser maior que o tamanho da validação ({len(self.timeseries.valid_target)}).")
                return False

            # --- Estágio de Montagem dos Parâmetros ---
            self.parameters = {
                "input_length": input_len,
                "output_length": output_len,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "save_checkpoints": save_checkpoints,
                "random_state": self.random_state,
            }
            return True  # Sucesso

        except (ValueError, TypeError):
            messagebox.showerror("Erro de Formato", "Por favor, insira apenas valores numéricos válidos nos campos.")
            return False
        except Exception as e:
            messagebox.showerror("Erro Inesperado", f"Ocorreu um erro: {e}")
            return False

class NModelWindow(BasicWindow):
    def __init__(self,series, index, remaining_models, *args, **kwargs):
        super().__init__(series, index, remaining_models, *args, **kwargs)

        # Random State
        self.random_state = 42

        # Input Chunk Length
        self.label_input_chunck_length = ctk.CTkLabel(master=self.hiperparameter_frame, text="Input Chunck Length:", font=("Arial", 14))
        self.label_input_chunck_length.grid(row=0, column=0, padx=5, pady=5)
        self.entry_input_chunck_length = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_input_chunck_length.grid(row=0, column=1, padx=5, pady=5)

        # Output Chunk Length
        self.label_output_chunck_length = ctk.CTkLabel(master=self.hiperparameter_frame, text="Output Chunck Length:", font=("Arial", 14))
        self.label_output_chunck_length.grid(row=1, column=0, padx=5, pady=5)
        self.entry_output_chunck_length = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_output_chunck_length.grid(row=1, column=1, padx=5, pady=5)

        # Number of Stacks
        self.label_num_stacks = ctk.CTkLabel(master=self.hiperparameter_frame, text="Number of Stacks:", font=("Arial", 14))
        self.label_num_stacks.grid(row=2, column=0, padx=5, pady=5)
        self.entry_num_stacks = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_num_stacks.grid(row=2, column=1, padx=5, pady=5)

        # Number of Blocks
        self.label_num_blocks = ctk.CTkLabel(master=self.hiperparameter_frame, text="Number of Blocks:", font=("Arial", 14))
        self.label_num_blocks.grid(row=3, column=0, padx=5, pady=5)
        self.entry_num_blocks = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_num_blocks.grid(row=3, column=1, padx=5, pady=5)

        # Number of Layers
        self.label_num_layers = ctk.CTkLabel(master=self.hiperparameter_frame, text="Number of Layers:", font=("Arial", 14))
        self.label_num_layers.grid(row=4, column=0, padx=5, pady=5)
        self.entry_num_layers = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_num_layers.grid(row=4, column=1, padx=5, pady=5)

        # Layer Width
        self.label_layer_widths = ctk.CTkLabel(master=self.hiperparameter_frame, text="Layer Width:", font=("Arial", 14))
        self.label_layer_widths.grid(row=0, column=2, padx=5, pady=5)
        self.entry_layer_widths = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_layer_widths.grid(row=0, column=3, padx=5, pady=5)

        # Dropout
        self.label_dropout = ctk.CTkLabel(master=self.hiperparameter_frame, text="Dropout:", font=("Arial", 14))
        self.label_dropout.grid(row=1, column=2, padx=5, pady=5)
        self.entry_dropout = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_dropout.grid(row=1, column=3, padx=5, pady=5)

        # Activation
        self.label_activation = ctk.CTkLabel(master=self.hiperparameter_frame, text="Activation:", font=("Arial", 14))
        self.label_activation.grid(row=2, column=2, padx=5, pady=5)
        self.activation_functions = ['ReLU', 'RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid']
        self.option_activation = ctk.CTkOptionMenu(master=self.hiperparameter_frame, values=self.activation_functions, state="disabled")
        self.option_activation.set("Selecione")
        self.option_activation.grid(row=2, column=3, padx=5, pady=5)

        # Batch Size
        self.label_batch_size = ctk.CTkLabel(master=self.hiperparameter_frame, text="Batch Size:", font=("Arial", 14))
        self.label_batch_size.grid(row=3, column=2, padx=5, pady=5)
        self.batch_sizes = ['16', '32', '64', '128', '256', '512', '1024']
        self.option_batch_size = ctk.CTkOptionMenu(master=self.hiperparameter_frame, values=self.batch_sizes, state="disabled")
        self.option_batch_size.set("Selecione")
        self.option_batch_size.grid(row=3, column=3, padx=5, pady=5)

        # Number of Epochs
        self.label_n_epochs = ctk.CTkLabel(master=self.hiperparameter_frame, text="Num Epochs:", font=("Arial", 14))
        self.label_n_epochs.grid(row=4, column=2, padx=5, pady=5)
        self.n_epochs = ['100', '200', '300', '400', '500', '1000']
        self.option_n_epochs = ctk.CTkComboBox(master=self.hiperparameter_frame, values=self.n_epochs, state="disabled")
        self.option_n_epochs.set("Selecione/Digite")
        self.option_n_epochs.grid(row=4, column=3, padx=5, pady=5)

        # Save Checkpoints
        self.label_save_checkpoints = ctk.CTkLabel(master=self.hiperparameter_frame, text="Save Checkpoints:",
                                                   font=("Arial", 14))
        self.label_save_checkpoints.grid(row=2, column=4, padx=5, pady=5)
        self.boolean_options = ['True', 'False']
        self.option_save_checkpoints = ctk.CTkOptionMenu(master=self.hiperparameter_frame, values=self.boolean_options,
                                                         state="disabled")
        self.option_save_checkpoints.set("Selecione")
        self.option_save_checkpoints.grid(row=2, column=5, padx=5, pady=5)

        Tooltip(self.label_input_chunck_length,text="Tamanho da entrada de dados(número).")
        Tooltip(self.label_output_chunck_length, text="Tamanho da saída de dados(número).")
        Tooltip(self.label_num_stacks, text="Quantidade de Stacks que constituem o modelo(número).")
        Tooltip(self.label_num_blocks, text="Quantidade de blocos que constituem uma Stack(número).")
        Tooltip(self.label_num_layers, text="Quantidade de camadas totalmente conectadas que precedem a camada final e que compõem cada bloco de cada Stack(número).")
        Tooltip(self.label_layer_widths, text="Determina o número de neurônios que compõem cada camada totalmente conectada em cada bloco de cada pilha. \n "
                                             "Se uma lista for passada, ela deve ter um comprimento igual a num_stacks, e cada entrada dessa lista corresponderá \n"
                                             "à largura das camadas da pilha correspondente. Se um número inteiro for passado, todas as pilhas terão blocos com \n"
                                             "camadas totalmente conectadas da mesma largura. (União[números, Lista[números]])")
        Tooltip(self.label_dropout, text="A probabilidade de dropout a ser utilizada nas camadas totalmente conectadas.")
        Tooltip(self.label_activation, text="Função de ativação utilizada no modelo.")
        Tooltip(self.label_batch_size, text="Número de séries temporais (sequências de entrada e saída) utilizadas em cada passagem de treinamento.")
        Tooltip(self.label_n_epochs, text="Número de épocas em cada rodada de treinamento do modelo.")
        Tooltip(self.label_save_checkpoints, text="Define se o modelo não treinado e os checkpoints do treinamento serão salvos automaticamente.")


class NBEATSModelWindow(NModelWindow):
    def __init__(self, series, index, remaining_models, *args, **kwargs):
        super().__init__(series, index, remaining_models, *args, **kwargs)
        self.title("N-BEATS - Escolha os Parâmetros")
        self.centralize_window(1070,200)

        #Expansion Coefficient Dimensionality
        self.label_expansion_coefficient_dim = ctk.CTkLabel(master=self.hiperparameter_frame, text="Expansion Coeff Dim:", font=("Arial", 14))
        self.label_expansion_coefficient_dim.grid(row=0, column=4, padx=5, pady=5)
        self.entry_expansion_coefficient_dim = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_expansion_coefficient_dim.grid(row=0, column=5, padx=5, pady=5)

        self.label_save_checkpoints.grid(row=1, column=4, padx=5, pady=5)
        self.option_save_checkpoints.grid(row=1, column=5, padx=5, pady=5)

        Tooltip(self.label_expansion_coefficient_dim,text="Dimensionalidade dos coeficientes de expansão — controla quantos parâmetros a rede usa para representar padrões como tendência e sazonalidade.\n"
                                 "Valores mais altos capturam maior complexidade, mas podem aumentar o risco de overfitting. Valores Inteiros. Recomenda-se de 3 a 7.")

    def disable_parameters(self):
        self.entry_input_chunck_length.configure(state="disabled")
        self.entry_output_chunck_length.configure(state="disabled")
        self.entry_num_stacks.configure(state="disabled")
        self.entry_num_blocks.configure(state="disabled")
        self.entry_num_layers.configure(state="disabled")
        self.entry_layer_widths.configure(state="disabled")
        self.entry_dropout.configure(state="disabled")
        self.option_activation.configure(state="disabled")
        self.option_batch_size.configure(state="disabled")
        self.option_n_epochs.configure(state="disabled")
        self.entry_expansion_coefficient_dim.configure(state="disabled")
        self.option_save_checkpoints.configure(state="disabled")

    def enable_parameters(self):
        self.entry_input_chunck_length.configure(state="normal")
        self.entry_output_chunck_length.configure(state="normal")
        self.entry_num_stacks.configure(state="normal")
        self.entry_num_blocks.configure(state="normal")
        self.entry_num_layers.configure(state="normal")
        self.entry_layer_widths.configure(state="normal")
        self.entry_dropout.configure(state="normal")
        self.option_activation.configure(state="normal")
        self.option_batch_size.configure(state="normal")
        self.option_n_epochs.configure(state="normal")
        self.entry_expansion_coefficient_dim.configure(state="normal")
        self.option_save_checkpoints.configure(state="normal")

    def confg_event(self, choice): #A DEFINIR ESCOLHAS DE VALORES AINDA
        if choice == 'Opção 1':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0,"18")
            self.entry_output_chunck_length.insert(0,"6")
            self.entry_num_stacks.insert(0,"3")
            self.entry_num_blocks.insert(0,"4")
            self.entry_num_layers.insert(0,"3")
            self.entry_layer_widths.insert(0,"5")
            self.entry_dropout.insert(0,"0.3")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("100")
            self.entry_expansion_coefficient_dim.insert(0,"5")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 2':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "18")
            self.entry_output_chunck_length.insert(0, "6")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "0.3")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("750")
            self.entry_expansion_coefficient_dim.insert(0,"3")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 3':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "18")
            self.entry_output_chunck_length.insert(0, "6")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "0.3")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("1000")
            self.entry_expansion_coefficient_dim.insert(0,"7")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Manual':
            self.clear_button.configure(state="normal")
            self.enable_parameters()
            self.clean_parameters()

    def clean_parameters(self):
        self.entry_input_chunck_length.delete(0,"end")
        self.entry_output_chunck_length.delete(0,"end")
        self.entry_num_stacks.delete(0,"end")
        self.entry_num_blocks.delete(0,"end")
        self.entry_num_layers.delete(0,"end")
        self.entry_layer_widths.delete(0,"end")
        self.entry_dropout.delete(0,"end")
        self.option_activation.set("Selecione")
        self.option_batch_size.set("Selecione")
        self.option_n_epochs.set("Selecione/Digite")
        self.entry_expansion_coefficient_dim.delete(0, "end")
        self.option_save_checkpoints.set("Selecione")

    # Versão corrigida e mais robusta do seu método

    def get_parameters(self):
        # --- Pega os valores dos widgets ---
        input_len_str = self.entry_input_chunck_length.get()
        output_len_str = self.entry_output_chunck_length.get()
        num_stacks_str = self.entry_num_stacks.get()
        num_blocks_str = self.entry_num_blocks.get()
        num_layers_str = self.entry_num_layers.get()
        layer_widths_str = self.entry_layer_widths.get()
        n_epochs_str = self.option_n_epochs.get()
        dropout_str = self.entry_dropout.get()
        activation_str = self.option_activation.get()
        batch_size_str = self.option_batch_size.get()
        expansion_coefficient_dim_str = self.entry_expansion_coefficient_dim.get()
        save_checkpoints_str = self.option_save_checkpoints.get()
        # --- Estágio de Validação (Antes de converter) ---
        # Verifica se todos os campos estão preenchidos
        all_fields = {
            "Input Chunk Length": input_len_str,
            "Output Chunk Length": output_len_str,
            "Num Stacks": num_stacks_str,
            "Num Blocks": num_blocks_str,
            "Num Layers": num_layers_str,
            "Layer Widths": layer_widths_str,
            "Nº Epochs": n_epochs_str,
            "Dropout": dropout_str,
            "Expansion Coeff Dim": expansion_coefficient_dim_str,
        }

        for field_name, value in all_fields.items():
            print(field_name, value, type(value))
            if not value:  # Checa se o campo está vazio
                messagebox.showerror("Erro de Validação", f"O campo '{field_name}' não pode estar vazio.")
                return False

        # Verifica os campos de seleção
        if activation_str == "Selecione":
            messagebox.showerror("Erro de Validação", "Por favor, selecione uma 'Activation Function'.")
            return False
        if batch_size_str == "Selecione":
            messagebox.showerror("Erro de Validação", "Por favor, selecione um 'Batch Size'.")
            return False
        if n_epochs_str == "Selecione/Digite":
            messagebox.showerror("Erro de Validação", "Por favor, selecione ou digite o número de 'Epochs'.")
            return False
        if save_checkpoints_str == "Selecione":
            messagebox.showerror("Erro de Validação", "Por favor, selecione uma opção em 'Save Checkpoints'.")
            return False

        # --- Estágio de Conversão e Lógica ---
        try:
            # Agora a conversão é mais segura
            input_len = int(input_len_str)
            output_len = int(output_len_str)
            num_stacks = int(num_stacks_str)
            num_blocks = int(num_blocks_str)
            num_layers = int(num_layers_str)
            layer_widths = int(layer_widths_str)
            n_epochs = int(n_epochs_str)
            dropout = float(dropout_str)
            batch_size = int(batch_size_str)
            expansion_coefficient_dim = int(expansion_coefficient_dim_str)
            save_checkpoints = save_checkpoints_str == 'true'  # Converte para boolean

            # ... (O resto da sua lógica de validação de valores negativos, etc., vai aqui) ...
            # Exemplo:
            if input_len < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Input Chunk Length não pode ser negativo: '{input_len_str}'")
                return False
            if input_len < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Input Chunk Length não pode ser negativo: '{input_len_str}'")
                return False  # Stop execution

            if output_len < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Output Chunk Length não pode ser negativo: '{output_len_str}'")
                return False  # Stop execution

            if (input_len + output_len) > len(self.timeseries.valid_target):
                messagebox.showerror("Erro de Validação",
                                     f"A soma de Input ({input_len}) e Output ({output_len}) não pode ser maior que o tamanho da validação ({len(self.timeseries.valid_target)}).")
                return False
            if num_stacks < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Number of Stacks não pode ser negativo: '{num_stacks_str}'")
                return False  # Stop execution

            if num_blocks < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Number of Blocks não pode ser negativo: '{num_blocks_str}'")
                return False  # Stop execution

            if num_layers < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Number of Layers não pode ser negativo: '{num_layers_str}'")
                return False  # Stop execution

            if layer_widths < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Layer Widths não pode ser negativo: '{layer_widths_str}'")
                return False  # Stop execution

            if n_epochs < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Number of Epochs não pode ser negativo: '{n_epochs}'")
                return False  # Stop execution

            if batch_size < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Batch Size não pode ser negativo: '{batch_size_str}'")
                return False  # Stop execution

            if expansion_coefficient_dim < 0:
                messagebox.showerror("Erro de Validação",
                                     f"Expansion Coeff Dim não pode ser negativo: '{expansion_coefficient_dim_str}'")
                return False  # Stop execution

            # --- Estágio de Montagem dos Parâmetros ---
            self.parameters = {
                "input_chunk_length": input_len,
                "output_chunk_length": output_len,
                "num_stacks": num_stacks,
                "num_blocks": num_blocks,
                "num_layers": num_layers,
                "layer_widths": layer_widths,
                "n_epochs": n_epochs,
                "random_state": self.random_state,
                "dropout": dropout,
                "activation": activation_str,  # Usa a string original
                "batch_size": batch_size,
                "expansion_coefficient_dim": expansion_coefficient_dim,
                "save_checkpoints": save_checkpoints
            }

            return True  # Sucesso

        except (ValueError, TypeError):
            # Este 'except' agora vai pegar apenas erros de digitação (ex: "abc" em um campo numérico)
            messagebox.showerror("Erro de Formato",
                                 "Por favor, insira apenas valores numéricos válidos nos campos.")
            return False
        except Exception as e:
            messagebox.showerror("Erro Inesperado", f"Ocorreu um erro: {e}")
            return False

class NHiTSModelWindow(NModelWindow):
    @staticmethod
    def get_option(option_value):
        try:
            return list(map(int, ast.literal_eval(option_value)))
        except Exception:
            return None

    def __init__(self, series, index, remaining_models, *args, **kwargs):
        super().__init__(series, index, remaining_models, *args, **kwargs)
        self.title("N-HiTS - Escolha os Parâmetros")
        self.centralize_window(1090,200)

        # Pooling Kernel Sizes
        self.label_pooling_kernel_sizes = ctk.CTkLabel(master=self.hiperparameter_frame, text="Pooling Kernel Sizes:", font=("Arial", 14))
        self.label_pooling_kernel_sizes.grid(row=0, column=4, padx=5, pady=5)
        self.pooling_kernel_sizes = ["Nenhum","[3, 5]","[5, 7, 9]","[3, 3, 3]"]
        self.options_pooling_kernel_sizes = ctk.CTkComboBox(master=self.hiperparameter_frame, values=self.pooling_kernel_sizes, state="disabled")
        self.options_pooling_kernel_sizes.set("Selecione/Digite")
        self.options_pooling_kernel_sizes.grid(row=0, column=5, padx=5, pady=5)

        # Interpolation Mode
        self.label_n_freq_downsample = ctk.CTkLabel(master=self.hiperparameter_frame, text="Num Freq Downsample:", font=("Arial", 14))
        self.label_n_freq_downsample.grid(row=1, column=4, padx=5, pady=5)
        self.n_freq_downsample = ["Nenhum","[1, 2]","[1, 1, 2]","[1, 2, 4]"]
        self.option_n_freq_downsample = ctk.CTkComboBox(master=self.hiperparameter_frame, values=self.n_freq_downsample, state="disabled")
        self.option_n_freq_downsample.set("Selecione/Digite")
        self.option_n_freq_downsample.grid(row=1, column=5, padx=5, pady=5)

        Tooltip(self.label_pooling_kernel_sizes, text="Tamanhos dos kernels usados nas operações de pooling dentro do modelo. Pooling é uma operação\n"
                                                      "que reduz a dimensão temporal da série mantendo as características mais importantes.\n"
                                                      "Formato tupla[tupla[inteiro]]. Recomenda-se valores de 2 a 10.")
        Tooltip(self.label_n_freq_downsample, text="Lista de inteiros que definem o fator de downsampling aplicado na frequência da série para diferentes\n"
                                                   "stacks do modelo, basicamente reduzindo a resolução temporal dos dados processados em cada nível.\n"
                                                   "Formato tupla[tupla[inteiro]]. Recomenda-se valores de 1 a 4.")

    def disable_parameters(self):
        self.entry_input_chunck_length.configure(state="disabled")
        self.entry_output_chunck_length.configure(state="disabled")
        self.entry_num_stacks.configure(state="disabled")
        self.entry_num_blocks.configure(state="disabled")
        self.entry_num_layers.configure(state="disabled")
        self.entry_layer_widths.configure(state="disabled")
        self.entry_dropout.configure(state="disabled")
        self.option_activation.configure(state="disabled")
        self.option_batch_size.configure(state="disabled")
        self.option_n_epochs.configure(state="disabled")
        self.options_pooling_kernel_sizes.configure(state="disabled")
        self.option_n_freq_downsample.configure(state="disabled")
        self.option_save_checkpoints.configure(state="disabled")

    def enable_parameters(self):
        self.entry_input_chunck_length.configure(state="normal")
        self.entry_output_chunck_length.configure(state="normal")
        self.entry_num_stacks.configure(state="normal")
        self.entry_num_blocks.configure(state="normal")
        self.entry_num_layers.configure(state="normal")
        self.entry_layer_widths.configure(state="normal")
        self.entry_dropout.configure(state="normal")
        self.option_activation.configure(state="normal")
        self.option_batch_size.configure(state="normal")
        self.option_n_epochs.configure(state="normal")
        self.options_pooling_kernel_sizes.configure(state="normal")
        self.option_n_freq_downsample.configure(state="normal")
        self.option_save_checkpoints.configure(state="normal")

    def confg_event(self, choice):  # A DEFINIR ESCOLHAS DE VALORES AINDA
        if choice == 'Opção 1':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "18")
            self.entry_output_chunck_length.insert(0, "6")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "0.3")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("100")
            self.options_pooling_kernel_sizes.set("Nenhum")
            self.option_n_freq_downsample.set("Nenhum")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 2':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "18")
            self.entry_output_chunck_length.insert(0, "6")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "0.3")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("750")
            self.options_pooling_kernel_sizes.set("[3, 5]")
            self.option_n_freq_downsample.set("[1, 2]")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 3':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "18")
            self.entry_output_chunck_length.insert(0, "6")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "0.3")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("1000")
            self.options_pooling_kernel_sizes.set("[3, 3, 3]")
            self.option_n_freq_downsample.set("[1, 2, 4]")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Manual':
            self.clear_button.configure(state="normal")
            self.enable_parameters()
            self.clean_parameters()

    def clean_parameters(self):
        self.entry_input_chunck_length.delete(0,"end")
        self.entry_output_chunck_length.delete(0,"end")
        self.entry_num_stacks.delete(0,"end")
        self.entry_num_blocks.delete(0,"end")
        self.entry_num_layers.delete(0,"end")
        self.entry_layer_widths.delete(0,"end")
        self.entry_dropout.delete(0,"end")
        self.option_activation.set("Selecione")
        self.option_batch_size.set("Selecione")
        self.option_n_epochs.set("Selecione/Digite")
        self.options_pooling_kernel_sizes.set("Selecione/Digite")
        self.option_n_freq_downsample.set("Selecione/Digite")
        self.option_save_checkpoints.set("Selecione")

    def get_parameters(self):
        # --- Pega os valores dos widgets ---
        input_len_str = self.entry_input_chunck_length.get()
        output_len_str = self.entry_output_chunck_length.get()
        num_stacks_str = self.entry_num_stacks.get()
        num_blocks_str = self.entry_num_blocks.get()
        num_layers_str = self.entry_num_layers.get()
        layer_widths_str = self.entry_layer_widths.get()
        dropout_str = self.entry_dropout.get()
        activation_str = self.option_activation.get()
        batch_size_str = self.option_batch_size.get()
        n_epochs_str = self.option_n_epochs.get()
        pooling_kernel_sizes_str = self.options_pooling_kernel_sizes.get()
        n_freq_downsample_str = self.option_n_freq_downsample.get()
        save_checkpoints_str = self.option_save_checkpoints.get()

        # --- Estágio de Validação (Antes de converter) ---
        all_fields = {
            "Input Chunk Length": input_len_str,
            "Output Chunk Length": output_len_str,
            "Num Stacks": num_stacks_str,
            "Num Blocks": num_blocks_str,
            "Num Layers": num_layers_str,
            "Layer Widths": layer_widths_str,
            "Dropout": dropout_str,
        }
        for field_name, value in all_fields.items():
            if not value:
                messagebox.showerror("Erro de Validação", f"O campo '{field_name}' não pode estar vazio.")
                return False

        # Verifica os campos de seleção
        if activation_str == "Selecione":
            messagebox.showerror("Erro de Validação", "Por favor, selecione uma 'Activation Function'.")
            return False
        if batch_size_str == "Selecione":
            messagebox.showerror("Erro de Validação", "Por favor, selecione um 'Batch Size'.")
            return False
        if n_epochs_str == "Selecione/Digite":
            messagebox.showerror("Erro de Validação", "Por favor, selecione ou digite o número de 'Epochs'.")
            return False
        if pooling_kernel_sizes_str == "Selecione/Digite":
            messagebox.showerror("Erro de Validação", "Por favor, selecione ou digite o 'Pooling Kernel Sizes'.")
            return False
        if n_freq_downsample_str == "Selecione/Digite":
            messagebox.showerror("Erro de Validação", "Por favor, selecione ou digite o 'Num Freq Downsample'.")
            return False
        if save_checkpoints_str == "Selecione":
            messagebox.showerror("Erro de Validação", "Por favor, selecione uma opção em 'Save Checkpoints'.")
            return False

        # --- Estágio de Conversão e Lógica ---
        try:
            input_len = int(input_len_str)
            output_len = int(output_len_str)
            num_stacks = int(num_stacks_str)
            num_blocks = int(num_blocks_str)
            num_layers = int(num_layers_str)
            layer_widths = int(layer_widths_str)
            dropout = float(dropout_str)
            batch_size = int(batch_size_str)
            n_epochs = int(n_epochs_str)
            save_checkpoints = save_checkpoints_str == 'True'

            pooling_kernel_sizes = self.get_option(pooling_kernel_sizes_str)
            if pooling_kernel_sizes is None and pooling_kernel_sizes_str.lower() != 'nenhum':
                messagebox.showerror("Erro de Formato",
                                     f"Formato inválido para 'Pooling Kernel Sizes': {pooling_kernel_sizes_str}")
                return False

            n_freq_downsample = self.get_option(n_freq_downsample_str)
            if n_freq_downsample is None and n_freq_downsample_str.lower() != 'nenhum':
                messagebox.showerror("Erro de Formato",
                                     f"Formato inválido para 'Num Freq Downsample': {n_freq_downsample_str}")
                return False

            # Validações lógicas
            if any(v <= 0 for v in
                   [input_len, output_len, num_stacks, num_blocks, num_layers, layer_widths, batch_size, n_epochs]):
                messagebox.showerror("Erro de Validação", "Valores de parâmetros numéricos devem ser maiores que zero.")
                return False

            if not (0.0 <= dropout < 1.0):
                messagebox.showerror("Erro de Validação", "O valor de 'Dropout' deve estar entre 0.0 e 0.99.")
                return False

            if (input_len + output_len) > len(self.timeseries.valid_target):
                messagebox.showerror("Erro de Validação",
                                     f"A soma de Input ({input_len}) e Output ({output_len}) não pode ser maior que o tamanho da validação ({len(self.timeseries.valid_target)}).")
                return False

            # --- Estágio de Montagem dos Parâmetros ---
            self.parameters = {
                "input_chunk_length": input_len,
                "output_chunk_length": output_len,
                "num_stacks": num_stacks,
                "num_blocks": num_blocks,
                "num_layers": num_layers,
                "layer_widths": layer_widths,
                "n_epochs": n_epochs,
                "random_state": self.random_state,
                "dropout": dropout,
                "activation": activation_str,
                "batch_size": batch_size,
                "pooling_kernel_sizes": pooling_kernel_sizes,
                "n_freq_downsample": n_freq_downsample,
                "save_checkpoints": save_checkpoints
            }
            return True  # Sucesso

        except (ValueError, TypeError):
            messagebox.showerror("Erro de Formato", "Por favor, insira apenas valores numéricos válidos nos campos.")
            return False
        except Exception as e:
            messagebox.showerror("Erro Inesperado", f"Ocorreu um erro: {e}")
            return False


class DataWindow(ctk.CTkToplevel):
    def __init__(self, master, callback=None):
        super().__init__(master)
        self.grid_propagate(True)
        self.title("Dados - Upload")
        self.callback = callback  # função para retornar os valores
        self.prate_file=None
        self.flow_file=None


        #Arquivo da Máscara
        self.upload_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.upload_frame.pack(pady=(30,10))

        self.upload_prate_button = ctk.CTkButton(self.upload_frame, text="Série Precipitação", command=self.upload_prate)
        self.upload_prate_button.grid(row=0, column=0, padx=5)
        self.label_prate_file = ctk.CTkLabel(self.upload_frame, text="Nenhum arquivo selecionado")
        self.label_prate_file.grid(row=1, column=0, padx=5, pady=(0,10))
        Tooltip(self.upload_prate_button, text="Upload da série de precipitação da bacia a ser estudada. (Formato CSV)")

        self.upload_flow_button = ctk.CTkButton(self.upload_frame, text="Série Vazão", command=self.upload_flow)
        self.upload_flow_button.grid(row=3, column=0, padx=5, pady=(10,0))
        self.label_flow_file = ctk.CTkLabel(self.upload_frame, text="Nenhum arquivo selecionado")
        self.label_flow_file.grid(row=4, column=0, padx=5)
        Tooltip(self.upload_flow_button, text="Upload da série de vazão da bacia a ser estudada. (Formato CSV)")


        # Botão salvar
        self.save_button = ctk.CTkButton(self, text="Salvar", command=self.save_parameters)
        self.save_button.pack(pady=10)

        # Faz a janela modal (bloqueia interação com a principal)
        self.transient(master)
        self.grab_set()
        self.focus_set()
        self.centralize_window()

    def centralize_window(self, width=300,height=260):
        # window_width = round(self.winfo_width(),-1)
        # window_height = round(self.winfo_height(),-1)
        window_width = width
        window_height = height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width)//2,-1)
        y = round((screen_height - window_height)//2,-1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

    def upload_prate(self):
        # Abre diálogo para selecionar arquivo .txt
        self.prate_file = filedialog.askopenfilename(
            title="Selecione arquivo de precipitação:",
            filetypes=[("CSV files", "*.csv")]
        )
        if self.prate_file:
            self.label_prate_file.configure(text=self.prate_file.split("/")[-1])  # mostra apenas o nome do arquivo

    def upload_flow(self):
        # Abre diálogo para selecionar arquivo .txt
        self.flow_file = filedialog.askopenfilename(
            title="Selecione arquivo de vazão:",
            filetypes=[("CSV files", "*.csv")]
        )
        if self.flow_file:
            self.label_flow_file.configure(text=self.flow_file.split("/")[-1])  # mostra apenas o nome do arquivo

    def save_parameters(self):
        try:
            if not self.prate_file:
                messagebox.showerror("Erro - Ausência de Precipitação", "Por favor, selecione um arquivo .csv contendo a precipitação.")
                return
            if not self.flow_file:
                messagebox.showerror("Erro - Ausência de Vazão", "Por favor, selecione um arquivo .csv contendo a vazão.")
                return

            params = {
                "prate": self.prate_file,
                "flow": self.flow_file
            }

        except ValueError:
            messagebox.showerror("Erro", "Envie os arquivos necessários!")
            return

        # Retorna os valores para a Application via callback
        if self.callback:
            self.callback(params)

        self.destroy()  # fecha a janela