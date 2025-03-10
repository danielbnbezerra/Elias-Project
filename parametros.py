from models import *
import tkinter as tk
import customtkinter as ctk

#JANELA PRINCIPAL COM CHECKBOX EM VEZ DE OPTION MENU, janelas de parâmetros abrem para cada checkbox marcada.

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
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Sair do Aplicativo?")
        self.geometry("300x150")
        self.centralize_window()
        self.grab_set()  # Make this window modal (prevents interaction with the main window)

        label = ctk.CTkLabel(self, text="Tem certeza que deseja sair?")
        label.pack(pady=20)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=10)

        yes_button = ctk.CTkButton(button_frame, text="Sim", command=parent.quit)
        yes_button.pack(side="left", padx=10)

        no_button = ctk.CTkButton(button_frame, text="Não", command=self.destroy)
        no_button.pack(side="right", padx=10)

    def centralize_window(self):
        # window_width = round(self.winfo_width(),-1)
        # window_height = round(self.winfo_height(),-1)
        window_width = 300
        window_height = 150
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width)//2,-1)
        y = round((screen_height - window_height)//2,-1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

class BasicWindow(ctk.CTkToplevel):
    def __init__(self, file, index, remaining_models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grab_set()
        self.grid_propagate(True)
        self.index = index
        self.selected_models = remaining_models
        self.model = None
        self.file = file
        self.parameters = None
        self.option_frame = ctk.CTkFrame(self, fg_color="#DDDDDD")
        self.option_frame.place(x=0, y=0, relwidth=0.2, relheight=1)
        self.option_frame.columnconfigure(0, weight=1)
        self.hiperparameter_frame = ctk.CTkFrame(self, fg_color="#EBEBEB")
        self.hiperparameter_frame.place(relx=0.2, y=0, relwidth=0.8, relheight=1)
        self.configurations= []

        #Tratamento de Dados
        self.checkbox_data_treat= ctk.CTkCheckBox(master=self.option_frame,
                                                  text="Tratar Dados",
                                                  font=("Arial",14))
        self.checkbox_data_treat.grid(row=0, column=0, padx=5, pady=5)

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
        print(self.index >= (len(self.selected_models) - 1))
        if self.index >= (len(self.selected_models) - 1):
            self.confirm_button = ctk.CTkButton(master=self.option_frame, text="Confirmar", command=self.model_run)
            self.confirm_button.grid(row=4, column=0, pady=10)
            Tooltip(self.confirm_button, text="Confirmar escolhas e executar o modelo.")
        else:
            self.next_button = ctk.CTkButton(master=self.option_frame, text="Próximo", command=self.next_window)
            self.next_button.grid(row=4, column=0, pady=10)
            Tooltip(self.next_button, text="Passa para a próxima janela de modelo a ser confiogurada.")

        Tooltip(self.label_confg_buttons, text="Selecione uma configuração predeterminada\n "
                                               "para o modelo ou edite manualmente os \n"
                                               "parâmetros:")
        Tooltip(self.clear_button, text="Limpar todos as escolhas de hiperparâmetros.")
    def get_configurations(self):
        self.get_parameters()
        self.configurations.append({"model":self.selected_models[self.index]["name"],"parameters":self.parameters})

    def next_window(self):
        # Fecha a janela atual e abre a próxima
        self.get_configurations()
        next_model_window = self.selected_models[self.index]["window"]
        next_model_window(self.file, self.index+1, self.selected_models)
        self.after(100, self.destroy)

    def centralize_window(self):
        # window_width = round(self.winfo_width(),-1)
        # window_height = round(self.winfo_height(),-1)
        window_width = 1040
        window_height = 240
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width)//2,-1)
        y = round((screen_height - window_height)//2,-1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

    def bring_fwd_window(self):
        self.attributes("-topmost", True)

    # def model_run(self):
    #     self.get_parameters()
    #     self.destroy()
    #     ModelRunWindow(self.parameters)
        # model_window = ctk.CTkToplevel(self)
        # model_window.title("Model Training")
        # model_window.geometry("400x300")
        #
        # # Progress bar for training
        # progress = ctk.CTkProgressBar(model_window)
        # progress.pack(pady=20)
        # progress.set(0.5)

    # def bottom_page_buttons(self):
    #     columns,rows= self.hiperparameter_frame.grid_size()  # Get current grid size
    #
    #     # "Limpar"
    #     self.clear_button = ctk.CTkButton(master=self.hiperparameter_frame, text="Limpar")
    #     self.clear_button.grid(row=rows, column=0, pady=10)
    #     Tooltip(self.clear_button, text="Limpar todos as escolhas de hiperparâmetros.")
    #
    #     # "Confirmar"
    #     self.confirm_button = ctk.CTkButton(master=self.hiperparameter_frame, text="Confirmar")
    #     self.confirm_button.grid(row=rows, column=columns-1, pady=10)
    #     Tooltip(self.confirm_button, text="Confirmar escolhas e executar o modelo.")


class LHCModelWindow(BasicWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("LHC - Escolha os Parâmetros")
        # self.model = LHCModel

    # def clean_parameters(self):
    #     self.entry_input_chunck_length.delete(0,"end")
    #     self.entry_output_chunck_length.delete(0,"end")
    #     self.entry_num_stacks.delete(0,"end")
    #     self.entry_num_blocks.delete(0,"end")
    #     self.entry_num_layers.delete(0,"end")
    #     self.entry_layer_width.delete(0,"end")
    #     self.entry_dropout.delete(0,"end")
    #     self.option_activation.set("Selecione")
    #     self.option_batch_size.set("Selecione")
    #     self.option_n_epoch.set("Selecione/Digite")
    #     self.option_save_checkpoint.set("Selecione")

    # def confg_event(self, choice): # A DEFINIR
    #     if choice == 'Opção 1':
    #
    #     if choice == 'Opção 2':
    #
    #     if choice == 'Opção 3':
    #
    #     if choice == 'Manual':
    #     return choice
    # def model_run(self):
    #     self.get_parameters()
    #     self.destroy()
    #     ModelRunLHCWindow(self.parameters)

class NModelWindow(BasicWindow):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

        # Random State
        self.random_state = 42

        # Input Chunk Length
        self.label_input_chunck_length = ctk.CTkLabel(master=self.hiperparameter_frame, text="Input Chunck Length:", font=("Arial", 14))
        self.label_input_chunck_length.grid(row=0, column=0, padx=5)
        self.entry_input_chunck_length = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_input_chunck_length.grid(row=0, column=1, padx=5)

        # Output Chunk Length
        self.label_output_chunck_length = ctk.CTkLabel(master=self.hiperparameter_frame, text="Output Chunck Length:", font=("Arial", 14))
        self.label_output_chunck_length.grid(row=1, column=0, padx=5)
        self.entry_output_chunck_length = ctk.CTkEntry(master=self.hiperparameter_frame, font=("Arial", 11), state="disabled")
        self.entry_output_chunck_length.grid(row=1, column=1, padx=5)

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

        #Save Checkpoints
        self.label_save_checkpoints = ctk.CTkLabel(master=self.hiperparameter_frame, text="Save Checkpoints:", font=("Arial", 14))
        self.label_save_checkpoints.grid(row=0, column=4, padx=5, pady=5)
        self.boolean_options = ['True', 'False']
        self.option_save_checkpoints = ctk.CTkOptionMenu(master=self.hiperparameter_frame, values=self.boolean_options, state="disabled")
        self.option_save_checkpoints.set("Selecione")
        self.option_save_checkpoints.grid(row=0, column=5, padx=5, pady=5)

        # self.bottom_page_buttons()

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("N-BEATS - Escolha os Parâmetros")
        print(self.file)
        # self.update_idletasks()
        # self.update()
        self.centralize_window()
        self.bring_fwd_window()

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
        self.option_save_checkpoints.configure(state="normal")

    def confg_event(self, choice): #A DEFINIR ESCOLHAS DE VALORES AINDA
        if choice == 'Opção 1':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0,"32")
            self.entry_output_chunck_length.insert(0,"21")
            self.entry_num_stacks.insert(0,"3")
            self.entry_num_blocks.insert(0,"4")
            self.entry_num_layers.insert(0,"3")
            self.entry_layer_widths.insert(0,"5")
            self.entry_dropout.insert(0,"2")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("500")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 2':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "32")
            self.entry_output_chunck_length.insert(0, "21")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "2")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("750")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 3':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "32")
            self.entry_output_chunck_length.insert(0, "21")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "2")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("1000")
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
        self.option_save_checkpoints.set("Selecione")

    def get_parameters(self):
        self.parameters = {
            "input_chunk_length": self.entry_input_chunck_length.get(),
            "output_chunk_length": self.entry_output_chunck_length.get(),
            "num_stacks": self.entry_num_stacks.get(),
            "num_blocks": self.entry_num_blocks.get(),
            "num_layers": self.entry_num_layers.get(),
            "layer_widths": self.entry_layer_widths.get(),
            "n_epochs": self.option_n_epochs.get(),
            "random_state": self.random_state,
            "dropout": self.entry_dropout.get(),
            "activation": self.option_activation.get(),
            "batch_size": self.option_batch_size.get(),
            "save_checkpoints": self.option_save_checkpoints.get()
        }

    def model_run(self):
        self.get_configurations()
        print(self.configurations)
        for i , model in enumerate(self.selected_models):
            if model["name"] == "N-BEATS":
                ModelRunNBEATSWindow(self.configurations[i]["parameters"])
        self.after(100, self.destroy)


class NHiTSModelWindow(NModelWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("N-HiTS - Escolha os Parâmetros")

        # self.update_idletasks()
        # self.update()
        self.centralize_window()
        self.bring_fwd_window()

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
        self.option_save_checkpoints.configure(state="normal")

    def confg_event(self, choice):  # A DEFINIR ESCOLHAS DE VALORES AINDA
        if choice == 'Opção 1':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "32")
            self.entry_output_chunck_length.insert(0, "21")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "2")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("500")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 2':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "32")
            self.entry_output_chunck_length.insert(0, "21")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "2")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("750")
            self.option_save_checkpoints.set("True")
            self.disable_parameters()

        if choice == 'Opção 3':
            self.clear_button.configure(state="disabled")
            self.enable_parameters()
            self.clean_parameters()
            self.entry_input_chunck_length.insert(0, "32")
            self.entry_output_chunck_length.insert(0, "21")
            self.entry_num_stacks.insert(0, "3")
            self.entry_num_blocks.insert(0, "4")
            self.entry_num_layers.insert(0, "3")
            self.entry_layer_widths.insert(0, "5")
            self.entry_dropout.insert(0, "2")
            self.option_activation.set("ReLU")
            self.option_batch_size.set("16")
            self.option_n_epochs.set("1000")
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
        self.option_save_checkpoints.set("Selecione")

    def get_parameters(self):
        self.parameters = {
            "input_chunk_length": self.entry_input_chunck_length.get(),
            "output_chunk_length": self.entry_output_chunck_length.get(),
            "num_stacks": self.entry_num_stacks.get(),
            "num_blocks": self.entry_num_blocks.get(),
            "num_layers": self.entry_num_layers.get(),
            "layer_widths": self.entry_layer_widths.get(),
            "n_epochs": self.option_n_epochs.get(),
            "random_state": self.random_state,
            "dropout": self.entry_dropout.get(),
            "activation": self.option_activation.get(),
            "batch_size": self.option_batch_size.get(),
            "save_checkpoints": self.option_save_checkpoints.get()
        }

    def model_run(self):
        self.get_configurations()
        for i, model in enumerate(self.selected_models):
            if model["name"] == "N-HiTS":
                ModelRunNHiTSWindow(self.configurations[i]["parameters"])
        self.after(100, self.destroy)