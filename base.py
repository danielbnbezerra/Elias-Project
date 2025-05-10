from parametros import *
import customtkinter as ctk
from tkinter import filedialog, messagebox

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Projeto Elias")
        self.centralize_window()
        self.protocol("WM_DELETE_WINDOW", self.confirm_exit)
        self.create_submenu()
        self.new_window = None
        self.model = None
        self.file = None
        self.possible_windows = [
            {"window": LHCModelWindow, "name":"LHC"},
            {"window": NBEATSModelWindow,"name":"N-BEATS"},
            {"window": NHiTSModelWindow,"name":"N-HiTS"}
        ]
        self.checkboxes = []

    # Configurações iniciais
        ctk.set_appearance_mode("light")  # "light", "dark", "system"
        ctk.set_default_color_theme("dark-blue")  # Temas: "blue", "green", "dark-blue"

    # Configuração inicial do layout
        self.grid_columnconfigure(0, weight=1)  # Configura a coluna 0 como central

    # Adicionando componentes

        #Frame de Opções
        self.main_options_frame = ctk.CTkFrame(self, fg_color="#DDDDDD")
        self.main_options_frame.place(x=0, y=0, relwidth=0.4, relheight=1)
        self.main_options_frame.columnconfigure(0, weight=1)
        self.main_options_frame.rowconfigure(0, weight=1)  # Espaço antes dos botões
        self.main_options_frame.rowconfigure(1, weight=1)

        predict_button = ctk.CTkButton(master= self.main_options_frame, text="Realizar Previsão", command=lambda: self.display_options("option_1"))
        predict_button.grid(row=0, column=0, padx=15, pady=10)

        #Frame de Opção Selecionada
        self.main_selected_options_frame = ctk.CTkFrame(self, fg_color="#EBEBEB")
        self.main_selected_options_frame.place(relx=0.4, y=0, relwidth=0.6, relheight=1)
        self.main_selected_options_frame.columnconfigure(0, weight=1)
        self.main_selected_options_frame.rowconfigure(0, weight=1)  # Espaço antes dos botões
        self.main_selected_options_frame.rowconfigure(5, weight=1)  # Espaço depois dos botões

        self.upload_button = None
        #Opção 1 - Realizar Previsão
        # self.models = None
        # self.model_option = None
        self.choose_model_button = None

        #Título Principal
        self.label_title = ctk.CTkLabel(master=self.main_selected_options_frame, text="Bem vindo ao Projeto Elias.", font=("Arial", 18))
        self.label_title.grid(row=0, column=0, pady=30)

    def display_options(self,option):

        #Destruindo opções existentes no menu direito.
        for widget in self.main_selected_options_frame.winfo_children():
            widget.destroy()

        if option == "option_1":
            #Seleção do Modelo
            for i, model in enumerate(self.possible_windows):
                var = ctk.BooleanVar()
                checkbox = ctk.CTkCheckBox(self.main_selected_options_frame, text=model["name"], variable=var)
                checkbox.grid(row=i+1, column=0, pady=5)
                self.checkboxes.append({"checkbox": checkbox, "var": var})

            #Confirmação
            self.choose_model_button = ctk.CTkButton(master=self.main_selected_options_frame, text="Confirmar", command=self.show_parameters)
            self.choose_model_button.grid(row=4, column=0, pady=10)

            #Upload
            self.upload_button = ctk.CTkButton(master=self.main_selected_options_frame, text="Upload", command=self.upload_file)
            Tooltip(self.upload_button, text="Upload do arquivo contendo os dados de entrada do modelo.")
            self.upload_button.grid(row=5, column=0, pady=30)

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

    # Criar Submenu
    def create_submenu(self):
        #Main Menu
        main_menu = tk.Menu(self)
        self.config(menu=main_menu)

        #File menu
        file_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Arquivo", menu=file_menu)
        file_menu.add_command(label="Upload Série", command=self.upload_file)
        # file_menu.add_command(label="Open", command=self.open_file)
        # file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.confirm_exit)

        # Edit menu
        # edit_menu = tk.Menu(main_menu, tearoff=0)
        # main_menu.add_cascade(label="Edit", menu=edit_menu)
        # edit_menu.add_command(label="Cut", command=self.cut)
        # edit_menu.add_command(label="Copy", command=self.copy)
        # edit_menu.add_command(label="Paste", command=self.paste)

        # View menu
        view_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Visualizar", menu=view_menu)
        view_menu.add_checkbutton(label="Tela Cheia", command=self.toggle_full_screen)
        # view_menu.add_command(label="Zoom In", command=self.zoom_in)
        # view_menu.add_command(label="Zoom Out", command=self.zoom_out)

        # Help menu
        help_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Ajuda", menu=help_menu)
        help_menu.add_command(label="Sobre", command=self.about)

    # File menu commands
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

    def show_parameters(self):
        # Pegando os modelos selecionados
        selected_models = [model for i, model in enumerate(self.possible_windows) if self.checkboxes[i]["var"].get()]

        if selected_models:
            # Criação da janela de parâmetros para o primeiro modelo selecionado
            self.parameter_window(selected_models, 0)

    def parameter_window(self, selected_models, index):
        if index > len(selected_models):
            print("Todos os modelos configurados!")
            return
        model_name_window = selected_models[index]["window"]
        if self.file:
            messagebox.showinfo("Arquivo Selecionado", f"Você selecionou o arquivo: {self.file}")
            if self.new_window is None or not self.new_window.winfo_exists():
                self.new_window = model_name_window(self.file, index+1, selected_models)
            else:
                self.new_window.focus()
        else:
            messagebox.showwarning("Nenhum arquivo selecionado", "Por favor, selecione um arquivo válido.")

    def centralize_window(self):
        window_width = 400
        window_height = 300
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")