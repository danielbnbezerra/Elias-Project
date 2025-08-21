import cfgrib
import os

# DEFINIR DIRETÓRIO DOS ARQUIVOS .grb2
dir_files = "DadosGFS//"

# CRIA LISTA COM O NOME DOS ARQUIVOS
file_list = []
for file in os.listdir(dir_files):
    if (file.split('.')[-1]=='grb2'):
        file_list.append(file)

for file in file_list:    
    datasets = cfgrib.open_datasets(dir_files+file)
    
    # IMPRIME NA TELA AS VARIÁVEIS DO CONJUNTO DE DADOS
    variable_dict = {}
    dataset_number = 0
    for dataset in datasets:
        variable_dict[dataset_number] = []
        for variable in dataset:
            variable_dict[dataset_number].append((variable, dataset[variable].attrs["long_name"], dataset[variable].attrs["units"]))
        dataset_number+=1
    
    f = open("lista_variaveis_gfs//"+file+"_variaveis.txt","a")
    f.write(file+"\n")
    for dataset_number in variable_dict.keys():
        f.write(str(dataset_number)+" DIMENSION NAMES "+str(list(datasets[dataset_number].dims))+"\n")
        for variable in variable_dict[dataset_number]:
            f.write(str(dataset_number)+" "+str(variable)+"\n")

    f.close()