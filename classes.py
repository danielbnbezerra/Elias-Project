#import models
import tkinter as tk
from doctest import master
from logging import disable
import customtkinter as ctk

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

class BasicWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model= None
        self.parameters= None
        self.grid_propagate(True)
        self.option_frame = ctk.CTkFrame(self, fg_color="#DDDDDD")
        self.option_frame.place(x=0, y=0, relwidth=0.2, relheight=1)
        self.hiperparameter_frame = ctk.CTkFrame(self, fg_color="#EBEBEB")
        self.hiperparameter_frame.place(relx=0.2, y=0, relwidth=0.8, relheight=1)

        #Configuração
        self.label_confg_buttons= ctk.CTkLabel(master=self.option_frame,
                                               text="Configuração:",
                                               font= ("Arial", 14))
        self.label_confg_buttons.grid(row=0, column=0, padx=5, pady=5)
        self.confgs_options = ["Opção 1", "Opção 2", "Opção 3", "Manual"]
        self.confgs_options = ctk.CTkOptionMenu(master=self.option_frame, values=self.confgs_options, command=self.confg_event)
        self.confgs_options.set("Selecione")
        self.confgs_options.grid(row=1, column=0, padx=5, pady=5)

        #Limpar
        self.clear_button = ctk.CTkButton(master=self.option_frame, text="Limpar", command=self.clean_parameters)
        self.clear_button.configure(state="disabled")
        self.clear_button.grid(row=2, column=0, pady=10)

        #Confirmar
        self.confirm_button = ctk.CTkButton(master=self.option_frame, text="Confirmar")  #,command=model_run)
        self.confirm_button.grid(row=3, column=0, pady=10)

        Tooltip(self.label_confg_buttons, text="Selecione uma configuração predeterminada\n "
                                               "para o modelo ou edite manualmente os \n"
                                               "parâmetros:")
        Tooltip(self.clear_button, text="Limpar todos as escolhas de hiperparâmetros.")
        Tooltip(self.confirm_button, text="Confirmar escolhas e executar o modelo.")

    def print_window_screen(self):
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        print(window_width, window_height, screen_width, screen_height)

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

    def model_run(self):
        # self.model.fit() #Preciso definir a série a se trabalhada ainda

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
        self.title("LHC Model")

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

class NModelWindow(BasicWindow):
    def __init__(self, *args, **kwargs):
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
        self.label_save_checkpoint = ctk.CTkLabel(master=self.hiperparameter_frame, text="Save Checkpoint:", font=("Arial", 14))
        self.label_save_checkpoint.grid(row=0, column=4, padx=5, pady=5)
        self.boolean_options = ['True', 'False']
        self.option_save_checkpoint = ctk.CTkOptionMenu(master=self.hiperparameter_frame, values=self.boolean_options, state="disabled")
        self.option_save_checkpoint.set("Selecione")
        self.option_save_checkpoint.grid(row=0, column=5, padx=5, pady=5)

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
        Tooltip(self.label_save_checkpoint, text="Define se o modelo não treinado e os checkpoints do treinamento serão salvos automaticamente.")


class NBEATSModelWindow(NModelWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("N-BEATS Model")
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
        self.option_save_checkpoint.configure(state="disabled")

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
        self.option_save_checkpoint.configure(state="normal")

    def get_parameters(self):
        params = {
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
            "save_checkpoint": self.option_save_checkpoint.get()
        }

        return params

    def model_construction(self):
        self.parameters = self.get_parameters()
        self.model = NBEATSModel(self.parameters)


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
            self.option_save_checkpoint.set("True")
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
            self.option_save_checkpoint.set("True")
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
            self.option_save_checkpoint.set("True")
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
        self.option_save_checkpoint.set("Selecione")


class NHiTSModelWindow(NModelWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("N-HiTS Model")

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
        self.option_save_checkpoint.configure(state="disabled")

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
        self.option_save_checkpoint.configure(state="normal")

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
            self.option_save_checkpoint.set("True")
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
            self.option_save_checkpoint.set("True")
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
            self.option_save_checkpoint.set("True")
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
        self.option_save_checkpoint.set("Selecione")