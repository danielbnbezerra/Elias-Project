import os

# DEFINIR DIRETÓRIO DOS ARQUIVOS .g2.tar
dir_arq = "DadosGFS"

# CRIA LISTA COM O NOME DOS ARQUIVOS
arq_lista = []
for arq in os.listdir(dir_arq):
    if (arq.split('.')[-1]=='tar'):
        arq_lista.append(arq)

# EXTRAI OS ARQUIVOS .grb2
for arq in arq_lista:
    arq_split = arq.split('_')
    nome = arq_split[0][:3]+"_"+arq_split[1]+"_"+arq_split[2][:-9]+"_0000_003.grb2"
    os.system("7z e "+dir_arq+"\\"+arq+" -o"+dir_arq+" "+nome)