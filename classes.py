import tkinter as tk
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
        self.grid_propagate(True)

    def print_window_screen(self):
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        print(window_width, window_height, screen_width, screen_height)

    def centralize_window(self):
        window_width = round(self.winfo_width(),-1)
        window_height = round(self.winfo_height(),-1)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width)//2,-1)
        y = round((screen_height - window_height)//2,-1)
        print(window_width,window_height,screen_width,screen_height,x,y)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")

    def bring_fwd_window(self):
        self.attributes("-topmost", True)

    def bottom_page_buttons(self):
        columns,rows= self.grid_size()  # Get current grid size
        print(rows,columns)
        # "Limpar" button at last row, first column
        self.clear_button = ctk.CTkButton(self, text="Limpar")
        self.clear_button.grid(row=rows, column=0, pady=10)
        # "Confirmar" button at last row, last column
        self.confirm_button = ctk.CTkButton(self, text="Confirmar")
        self.confirm_button.grid(row=rows, column=columns-1, pady=10)


class LHCModelWindow(BasicWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("LHC Model")

class NModelWindow(BasicWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        option_frame= ctk.CTkFrame(self,fg_color="lightblue")
        option_frame.place(x=0,y=0,relwidth=0.3,relheight=1)
        hiperparameter_frame = ctk.CTkFrame(self,fg_color="red")
        hiperparameter_frame.place(relx=0.3,y=0,relwidth=0.7,relheight=1)

        # Random State
        self.random_state = 42

        # Input Chunk Length
        self.label_input_chunck_length = ctk.CTkLabel(self, text="Input Chunck Length:", font=("Arial", 14))
        self.label_input_chunck_length.grid(row=0, column=0, padx=5)
        self.entry_input_chunck_length = ctk.CTkEntry(self, font=("Arial", 11))
        self.entry_input_chunck_length.grid(row=0, column=1, padx=5)

        # Output Chunk Length
        self.label_output_chunck_length = ctk.CTkLabel(self, text="Output Chunck Length:", font=("Arial", 14))
        self.label_output_chunck_length.grid(row=1, column=0, padx=5)
        self.entry_output_chunck_length = ctk.CTkEntry(self, placeholder_text="Tamanho output (número)",
                                                       font=("Arial", 11))
        self.entry_output_chunck_length.grid(row=1, column=1, padx=5)

        # Number of Stacks
        self.label_num_stacks = ctk.CTkLabel(self, text="Number of Stacks:", font=("Arial", 14))
        self.label_num_stacks.grid(row=2, column=0, padx=5, pady=20)
        self.entry_num_stacks = ctk.CTkEntry(self, placeholder_text="Qtd Stacks (número)", font=("Arial", 11))
        self.entry_num_stacks.grid(row=2, column=1, padx=5, pady=20)

        # Number of Blocks
        self.label_num_blocks = ctk.CTkLabel(self, text="Number of Blocks:", font=("Arial", 14))
        self.label_num_blocks.grid(row=3, column=0, padx=5, pady=20)
        self.entry_num_blocks = ctk.CTkEntry(self, placeholder_text="Qtd Blocos (número)", font=("Arial", 11))
        self.entry_num_blocks.grid(row=3, column=1, padx=5, pady=20)

        # Number of Layers
        self.label_num_layers = ctk.CTkLabel(self, text="Number of Layers:", font=("Arial", 14))
        self.label_num_layers.grid(row=4, column=0, padx=5, pady=20)
        self.entry_num_layers = ctk.CTkEntry(self, placeholder_text="Qtd Camadas (número)", font=("Arial", 11))
        self.entry_num_layers.grid(row=4, column=1, padx=5, pady=20)

        # Layer Width
        self.label_layer_width = ctk.CTkLabel(self, text="Layer Width:", font=("Arial", 14))
        self.label_layer_width.grid(row=0, column=2, padx=5, pady=20)
        self.entry_layer_width = ctk.CTkEntry(self, placeholder_text="Qtd Neurônios (União[números, Lista[números]])",
                                              font=("Arial", 11))
        self.entry_layer_width.grid(row=0, column=3, padx=5, pady=20)

        # Dropout
        self.label_dropout = ctk.CTkLabel(self, text="Dropout:", font=("Arial", 14))
        self.label_dropout.grid(row=1, column=2, padx=5, pady=20)
        self.entry_dropout = ctk.CTkEntry(self, placeholder_text="Qtd Dropout (número)", font=("Arial", 11))
        self.entry_dropout.grid(row=1, column=3, padx=5, pady=20)

        # Activation
        self.label_activation = ctk.CTkLabel(self, text="Activation:", font=("Arial", 14))
        self.label_activation.grid(row=2, column=2, padx=5, pady=20)
        self.activation_functions = ['ReLU', 'RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid']
        self.option_activation = ctk.CTkOptionMenu(self, values=self.activation_functions)
        self.option_activation.set("Função Ativação")
        self.option_activation.grid(row=2, column=3, padx=5, pady=20)

        # Batch Size
        self.label_batch_size = ctk.CTkLabel(self, text="Batch Size:", font=("Arial", 14))
        self.label_batch_size.grid(row=3, column=2, padx=5, pady=20)
        self.batch_sizes = ['16', '32', '64', '128', '256', '512', '1024']
        self.option_batch_size = ctk.CTkOptionMenu(self, values=self.batch_sizes)
        self.option_batch_size.set("Selecione um valor")
        self.option_batch_size.grid(row=3, column=3, padx=5, pady=20)

        # Number of Epochs
        self.label_n_epochs = ctk.CTkLabel(self, text="Number Epochs:", font=("Arial", 14))
        self.label_n_epochs.grid(row=4, column=2, padx=5, pady=20)
        self.n_epochs = ['100', '200', '300', '400', '500', '1000']
        self.option_n_epoch = ctk.CTkComboBox(self, values=self.n_epochs)
        self.option_n_epoch.set("Selecione/Digite")
        self.option_n_epoch.grid(row=4, column=3, padx=5, pady=20)

        #Save Checkpoints
        self.label_save_checkpoint = ctk.CTkLabel(self, text="Save Checkpoint:", font=("Arial", 14))
        self.label_save_checkpoint.grid(row=0, column=4, padx=5, pady=20)
        self.boolean_options = ['True', 'False']
        self.option_save_checkpoint = ctk.CTkOptionMenu(self, values=self.boolean_options)
        self.option_save_checkpoint.set("Selecionar Opção")
        self.option_save_checkpoint.grid(row=0, column=5, padx=5, pady=20)

        self.bottom_page_buttons()

        Tooltip(self.label_input_chunck_length,text="Tamanho da entrada de dados (número).")
        Tooltip(self.label_output_chunck_length, text="Tamanho da saída de dados (número).")
        Tooltip(self.label_input_chunck_length, text="Tamanho da entrada de dados (número).")
        Tooltip(self.label_input_chunck_length, text="Tamanho da entrada de dados (número).")


class NBEATSModelWindow(NModelWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("N-BEATS Model")

        self.update_idletasks()
        self.update()
        self.print_window_screen()
        self.centralize_window()
        self.bring_fwd_window()

class NHiTSModelWindow(NModelWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("N-HiTS Model")

        self.update_idletasks()
        self.update()
        self.print_window_screen()
        self.centralize_window()
        self.bring_fwd_window()