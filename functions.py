import customtkinter as ctk
from tkinter import filedialog, messagebox

# Função para abrir o diálogo de upload e limitar formatos específicos
def upload_file():
    # Filtrar formatos permitidos
    arquivo = filedialog.askopenfilename(
        title="Selecione um arquivo",
        filetypes=[
            ("Série Temporal", "*.CSV *.xslx *.NetCDF"),  # permite apenas imagens
        ]
    )
    # Verificar se o usuário selecionou um arquivo
    if arquivo:
        messagebox.showinfo("Arquivo Selecionado", f"Você selecionou: {arquivo}")
    else:
        messagebox.showwarning("Nenhum arquivo selecionado", "Por favor, selecione um arquivo válido.")

