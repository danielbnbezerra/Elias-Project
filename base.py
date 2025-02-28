from parametros import *
import customtkinter as ctk
from tkinter import filedialog, messagebox

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Projeto Elias")
        app_width= 400
        app_height= 300
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x= screen_width/2 - app_width/2
        y= screen_height/2 - app_height/2
        self.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)} ")
        self.create_submenu()
        self.param_window = None
        self.model = None
        self.file = None
        self.protocol("WM_DELETE_WINDOW", self.confirm_exit)

    # Configurações iniciais
        ctk.set_appearance_mode("light")  # "light", "dark", "system"
        ctk.set_default_color_theme("dark-blue")  # Temas: "blue", "green", "dark-blue"

    # Configuração inicial do layout
        self.grid_columnconfigure(0, weight=1)  # Configura a coluna 0 como central

    # Adicionando componentes

        #Título Principal
        self.label_title = ctk.CTkLabel(self, text="Bem vindo ao Projeto Elias.", font=("Arial", 20))
        self.label_title.grid(row=0, column=0, pady=30)

        #Seleção do Modelo
        self.models = ["LHC Model", "N-BEATS", "N-HiTS"]
        self.model_option = ctk.CTkOptionMenu(self, anchor="center", values=self.models)
        self.model_option.grid(row=1, column=0, pady=10)
        self.model_option.set("Escolha o modelo")  # Define a opção padrão

        #Confirmação
        self.choose_model_button = ctk.CTkButton(self, text="Confirmar", command=self.parameter_window)
        self.choose_model_button.grid(row=2, column=0, pady=10)

        #Upload
        self.upload_button = ctk.CTkButton(self, text="Upload", command=self.upload_file)
        Tooltip(self.upload_button, text="Upload do arquivo contendo os dados de entrada do modelo.")
        self.upload_button.grid(row=3, column=0, pady=30)

    def confirm_exit(self):
        ConfirmExitWindow(self)

    # Função para abrir o diálogo de upload e limitar formatos específicos
    def upload_file(self):
        # Filtrar formatos permitidos
        self.file = filedialog.askopenfilename(
            title="Selecione um arquivo",
            filetypes=[
                ("Dados", "*.csv *.xslx *.NetCDF"),  # permite apenas imagens
            ]
        )
        # Verificar se o usuário selecionou um arquivo
        # if arquivo:
        #     messagebox.showinfo("Arquivo Selecionado", f"Você selecionou: {arquivo}")
        # else:
        #     messagebox.showwarning("Nenhum arquivo selecionado", "Por favor, selecione um arquivo válido.")

    # Criar Submenu
    def create_submenu(self):
        #Main Menu
        main_menu = tk.Menu(self)
        self.config(menu=main_menu)

        #File menu
        file_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.confirm_exit)

        # Edit menu
        edit_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Cut", command=self.cut)
        edit_menu.add_command(label="Copy", command=self.copy)
        edit_menu.add_command(label="Paste", command=self.paste)

        # View menu
        view_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Full Screen", command=self.toggle_full_screen)
        view_menu.add_command(label="Zoom In", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.zoom_out)

        # Help menu
        help_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.about)

    # File menu commands
    def new_file(self):
        self.show_message("New File")

    def open_file(self):
        self.show_message("Open File")

    def save_file(self):
        self.show_message("Save File")

    def exit_app(self):
        self.quit()

    # Edit menu commands
    def cut(self):
        self.show_message("Cut")

    def copy(self):
        self.show_message("Copy")

    def paste(self):
        self.show_message("Paste")

    # View menu commands
    def toggle_full_screen(self):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def zoom_in(self):
        self.show_message("Info","Zoom In")

    def zoom_out(self):
        self.show_message("Info","Zoom Out")

    # Help menu commands
    def about(self):
        self.show_message("About","This is an example application with a menu!")

    # Utility function
    def show_message(self, title,msg):
        messagebox.showinfo(title, msg)

    def parameter_window(self):
        if self.file:
            messagebox.showinfo("Arquivo Selecionado", f"Você selecionou o arquivo: {self.file}")
            if self.param_window is None or not self.param_window.winfo_exists():
                if self.model_option.get() == self.models[0]:
                    self.param_window = LHCModelWindow(self.file)
                if self.model_option.get() == self.models[1]:
                    self.param_window = NBEATSModelWindow(self.file)
                if self.model_option.get() == self.models[2]:
                    self.param_window = NHiTSModelWindow(self.file)
            else:
                self.param_window.focus()
        else:
            messagebox.showwarning("Nenhum arquivo selecionado", "Por favor, selecione um arquivo válido.")

        # if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
        #     if self.model_option.get() == self.models[0]:
        #         self.toplevel_window = LHCModelWindow(self.file)
        #     if self.model_option.get() == self.models[1]:
        #         self.toplevel_window = NBEATSModelWindow(self.file)
        #     if self.model_option.get() == self.models[2]:
        #         self.toplevel_window = NHiTSModelWindow(self.file)
        # else:
        #     self.toplevel_window.focus()