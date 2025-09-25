from tkcalendar import DateEntry

from parametros import *
from series import *


class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Elias")
        self.centralize_window()
        self.protocol("WM_DELETE_WINDOW", self.confirm_exit)
        self.create_submenu()
        self.new_window = None
        self.model = None
        self.series_files = None
        self.data_intervals = []
        self.timeseries=None

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

        self.data_button = ctk.CTkButton(master= self.main_options_frame, text="Upload Dados", command=self.data_window)
        self.data_button.grid(row=0, column=0, padx=15, pady=10)
        Tooltip(self.data_button, text="Selecione as séries temporais da respectiva bacia hidrográfica.")

        #Frame de Opção Selecionada
        self.main_selected_options_frame = ctk.CTkFrame(self, fg_color="#EBEBEB")
        self.main_selected_options_frame.place(relx=0.4, y=0, relwidth=0.6, relheight=1)
        self.main_selected_options_frame.columnconfigure(0, weight=1)
        self.main_selected_options_frame.rowconfigure(0, weight=1)  # Espaço antes dos botões
        self.main_selected_options_frame.rowconfigure(5, weight=1)  # Espaço depois dos botões


        self.choose_model_button = None

        #Título Principal
        self.label_title_frame = ctk.CTkFrame(master= self.main_selected_options_frame, fg_color='transparent')
        self.label_title_frame.grid(row=0, column=0, pady=30)
        self.label_title = ctk.CTkLabel(master=self.label_title_frame, text="Bem vindo ao Elias!", font=("Arial", 18))
        self.label_title.grid(row=0, column=0, pady=10)
        self.label_subtitle = ctk.CTkLabel(master=self.label_title_frame, text="Importe os dados para começar.",
                                        font=("Arial", 14))
        self.label_subtitle.grid(row=1, column=0, pady=5)

    def display_options(self,option):

        #Destruindo opções existentes no menu direito.
        for widget in self.main_selected_options_frame.winfo_children():
            widget.destroy()

        if option == "display":
            #Seleção do Modelo

            # Intervalos
            self.start_date_frame = ctk.CTkFrame(self.main_selected_options_frame, fg_color="transparent")
            self.start_date_frame.pack(pady=5)
            self.label_start_date = ctk.CTkLabel(self.start_date_frame, text="Data Inicial:", font=("Arial", 14))
            self.label_start_date.grid(row=0, column=0, padx=5)
            self.entry_start_date = DateEntry(
                self.start_date_frame,
                date_pattern="dd/mm/yyyy",
                parent=self.start_date_frame,
                takefocus = False
            )
            self.entry_start_date.grid(row=0, column=1, padx=5)
            Tooltip(self.label_start_date, text="Escolha a data inicial dos dados a serem utilizados.")

            self.end_date_frame = ctk.CTkFrame(self.main_selected_options_frame, fg_color="transparent")
            self.end_date_frame.pack(pady=5)
            self.label_end_date = ctk.CTkLabel(self.end_date_frame, text="Data Final:", font=("Arial", 14))
            self.label_end_date.grid(row=0, column=0, padx=5)
            self.entry_end_date = DateEntry(
                self.end_date_frame,
                date_pattern="dd/mm/yyyy",
                parent=self.end_date_frame,
                takefocus = False
            )
            self.entry_end_date.grid(row=0, column=1, padx=5)
            Tooltip(self.label_end_date, text="Escolha a data final dos dados a serem utilizados.")

            self.train_valid_frame = ctk.CTkFrame(self.main_selected_options_frame, fg_color="transparent")
            self.train_valid_frame.pack(pady=5)
            self.label_train_valid = ctk.CTkLabel(master=self.train_valid_frame, text="Treino/Validação:", font=("Arial", 14))
            self.label_train_valid.grid(row=0, column=0, padx=5)
            self.train_valid = ['60/40', '65/35', '70/30', '80/20','85/15']
            self.option_train_valid = ctk.CTkComboBox(master=self.train_valid_frame, values=self.train_valid)
            self.option_train_valid.set("Selecione/Digite")
            self.option_train_valid.grid(row=0, column=1, padx=5)
            Tooltip(self.label_train_valid, text="Escolha a proporção de divisão dos dados entre treino e validação, soma deve representar 100%. Ex: '80/20'.")

            self.model_frame = ctk.CTkFrame(self.main_selected_options_frame, fg_color="transparent")
            self.model_frame.pack(pady=5)
            self.label_models = ctk.CTkLabel(self.model_frame, text="Modelos:", font=("Arial", 14))
            self.label_models.grid(row=0,column=0,pady=5)
            self.model_select_frame = ctk.CTkFrame(self.model_frame, fg_color="transparent")
            self.model_select_frame.grid(row=1, column=1, pady=5)
            for i, model in enumerate(self.possible_windows):
                var = ctk.BooleanVar()
                checkbox = ctk.CTkCheckBox(self.model_select_frame, text=model["name"], variable=var)
                checkbox.grid(row=i+1, column=0, pady=5)
                self.checkboxes.append({"checkbox": checkbox, "var": var})
            Tooltip(self.label_models, text="Escolha o modelo para realização do prognóstico.")

            #Confirmação
            self.choose_model_button = ctk.CTkButton(master=self.main_selected_options_frame, text="Confirmar", command=self.show_parameters)
            self.choose_model_button.pack(pady=5)
            Tooltip(self.choose_model_button, text="Confirma as escolhas e segue para escolha de parâmetros dos modelos selecionados.")

    # Criar Submenu
    def create_submenu(self):
        #Main Menu
        main_menu = tk.Menu(self)
        self.config(menu=main_menu)

        #File menu
        file_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Arquivo", menu=file_menu)
        file_menu.add_command(label="Upload", command=self.data_window)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.confirm_exit)

        # Help menu
        help_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Ajuda", menu=help_menu)
        help_menu.add_command(label="Sobre", command=self.about)

    # File menu commands
    def confirm_exit(self):
        ConfirmExitWindow(self)

    # Help menu commands
    def about(self):
        self.show_message("About","""MIT License

Copyright (c) 2025 Daniel Bezerra

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without 
restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following 
conditions:

The above copyright notice and this permission notice shall be 
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
OTHER DEALINGS IN THE SOFTWARE.""")

    # Utility function
    def show_message(self, title, msg):
        messagebox.showinfo(title, msg)

    def get_train_valid_values(self):
        try:
            parts = self.option_train_valid.get().split("/")
            if len(parts) != 2:
                raise ValueError("Formato errado.")
            try:
                train, valid = map(int, parts)
            except (ValueError, AttributeError):
                messagebox.showerror("Erro",
                                     f"Formato inválido: '{self.option_train_valid.get()}'. Preencha no formato 'treino/validação', soma deve representar 100%. Exemplo '80/20'.", parent=self)
                return False

            if train < 0 or valid < 0:
                messagebox.showerror(
                    "Erro",
                    f"Não podem existir valores negativos para treino ou validação. Valor atual: {train}/{valid}.", parent=self
                )
                return False
            # Verifica se a soma é 100
            if (train + valid) != 100:
                messagebox.showerror(
                    "Erro",
                    f"A soma de treino e validação deve ser 100. Valor atual: {train}/{valid}.", parent=self
                )
                return False
            train_percent = float(train) / 100
            return train_percent

        except (ValueError, AttributeError):
            messagebox.showerror("Erro",
                                 f"Formato inválido: '{self.option_train_valid.get()}'. Preencha no formato 'treino/validação', soma deve representar 100%. Exemplo '80/20'.", parent=self)
            return False

    def check_dates(self):
        train_percent = self.get_train_valid_values()
        if train_percent is False:
            return False
        self.timeseries = GetSeries(self.series_files, train_percent)
        try:
            # Pega as datas do DateEntry
            date_start = pd.to_datetime(self.entry_start_date.get(), dayfirst=True)
            date_end = pd.to_datetime(self.entry_end_date.get(), dayfirst=True)

            # Checa se a data inicial é menor que a final
            if date_start > date_end:
                messagebox.showerror("Erro", "A data inicial não pode ser maior que a data final.", parent=self)
                return False

            # Checa se a data inicial é igual a final
            if date_start == date_end:
                messagebox.showerror("Erro", "A data inicial não pode ser igual a data final.", parent=self)
                return False

            # Checa se as datas estão dentro da série temporal
            date_min_prate = self.timeseries.prate.start_time()
            date_max_prate = self.timeseries.prate.end_time()
            date_min_flow = self.timeseries.flow.start_time()
            date_max_flow = self.timeseries.flow.end_time()

            if not ((date_min_prate == date_min_flow) and (date_max_prate == date_max_flow)):
                messagebox.showerror("Erro", f"As datas iniciais e finais ({date_min_prate.strftime('%d/%m/%Y')} - {date_max_prate.strftime('%d/%m/%Y')}) da série de precipitação"
                                             f" não coincidem com as da série de vazão. ({date_min_flow.strftime('%d/%m/%Y')} - {date_max_flow.strftime('%d/%m/%Y')})", parent=self)
                return False

            #Pelo menos 30 valores para as séries carregadas, correspondendo a 25 valores após criação de séries derivadas
            if not (date_min_prate <= date_start <= (date_max_prate - pd.Timedelta(days=30))):
                messagebox.showerror("Erro", f"A data inicial da série deve estar entre {date_min_prate.strftime('%d/%m/%Y')} e {(date_max_prate- pd.Timedelta(days=30)).strftime('%d/%m/%Y')}.", parent=self)
                return False

            if not ((date_min_prate + pd.Timedelta(days=30)) <= date_end <= date_max_prate):
                messagebox.showerror("Erro", f"A data final da série deve estar entre {(date_min_prate + pd.Timedelta(days=30)).strftime('%d/%m/%Y')} e {date_max_prate.strftime('%d/%m/%Y')}.", parent=self)
                return False

            self.data_intervals.append(date_start)
            self.data_intervals.append(date_end)
            return True

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao validar datas: {e}", parent=self)
            return False

    def show_parameters(self):
        cleanup_darts_logs()
        # Pegando os modelos selecionados
        selected_models = [model for i, model in enumerate(self.possible_windows) if self.checkboxes[i]["var"].get()]
        if self.check_dates():
            self.timeseries.update_date_interval(self.data_intervals[0],self.data_intervals[1])
            if selected_models:
                # Criação da janela de parâmetros para o primeiro modelo selecionado
                self.parameter_window(selected_models, 0)
            else:
                messagebox.showerror("Erro - Selecione um Modelo",
                                     "Por favor, selecione pelo menos 1 modelo disponível.", parent=self)

    def data_window(self):
        def get_data_params(data_parameters):
            self.series_files = data_parameters
            messagebox.showinfo("Sucesso!", "Arquivos Carregados.", parent=self)
            self.display_options("display")

        # Abre a janela de parâmetros, passando a função como callback
        DataWindow(master=self, callback=get_data_params)

    def parameter_window(self, selected_models, index):
        if index > len(selected_models):
            self.show_message("Modelos",
                              "Modelos configurados com sucesso.\nComeçando treinamento.", parent=self)
            return
        model_name_window = selected_models[index]["window"]
        if self.new_window is None or not self.new_window.winfo_exists():
           self.new_window = model_name_window(self.timeseries, index+1, selected_models)
        else:
           self.new_window.focus()

    def centralize_window(self, width=480, height=380):
        window_width = width
        window_height = height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = round((screen_width - window_width) // 2, -1)
        y = round((screen_height - window_height) // 2, -1)
        self.geometry(f"{window_width}x{window_height}+{x}+{y} ")
