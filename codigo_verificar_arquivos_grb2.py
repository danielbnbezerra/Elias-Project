import os
from datetime import datetime, timedelta

# DEFINIR DIRETÓRIO DOS ARQUIVOS .g2.tar
dir_arq = "F:\\DadosGFS.grb2"

# CRIA LISTA COM AS DATAS DOS ARQUIVOS
data_lista = []
for arq in os.listdir(dir_arq):
    if (arq.split('.')[-1]=='grb2'):
        data_arq = arq.split("_")[2]
        data_arq = datetime(int(str(data_arq)[:4]), int(str(data_arq)[4:6]), int(str(data_arq)[6:8]))
        data_lista.append(data_arq)

# CRIA LISTA COM DATAS FALTANTES
datas_faltando = []
data_anterior = None
for data_arq in data_lista:
    if (data_anterior == None):
        data_anterior = data_arq
    else:
        if (data_arq != data_anterior + timedelta(days=1)):
            data = data_anterior + timedelta(days=1)
            while (data!=data_arq):
                datas_faltando.append(data)
                data += timedelta(days=1)
        data_anterior = data_arq

f = open("datas_faltantes_grb2.txt", "a")
for data_arq in datas_faltando:
    f.write(str(data_arq)+"\n")
f.close()