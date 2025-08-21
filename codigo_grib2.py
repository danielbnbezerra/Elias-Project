import cfgrib
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

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
    
    for dataset_number in variable_dict.keys():
        print (dataset_number, "DIMENSION NAMES", str(list(datasets[dataset_number].dims)))
        for variable in variable_dict[dataset_number]:
            print (dataset_number, variable)
            
    # DEFINIR VARIÁVEIS, PRESSÃO, LATITUDE E LONGITUDE
    
    date = file.split("_")[2]
    date = datetime.datetime(int(str(date)[:4]), int(str(date)[4:6]), int(str(date)[6:]))

    if (date >= datetime.datetime(2013, 1, 1) and date <= datetime.datetime(2015, 1, 14)):
        variable_list = [(41, ["prate"]), (9, ["t2m"]), (17, ["r"]) , (5, ["soilw"]), (8, ["u10", "v10"])]
    elif (date >= datetime.datetime(2015, 1, 15) and date <= datetime.datetime(2016, 5, 11)):
        variable_list = [(40, ["prate"]), (10, ["t2m"]), (18, ["r"]) , (6, ["soilw"]), (9, ["u10", "v10"])]
    elif (date >= datetime.datetime(2016, 5, 12) and date <= datetime.datetime(2019, 6, 12)):
        variable_list = [(42, ["prate"]), (10, ["t2m"]), (18, ["r"]) , (6, ["soilw"]), (9, ["u10", "v10"])]
    elif (date >= datetime.datetime(2019, 6, 13) and date <= datetime.datetime(2019, 11, 7)):
        variable_list = [(47, ["prate"]), (11, ["t2m"]), (19, ["r"]) , (6, ["soilw"]), (10, ["u10", "v10"])]
    elif (date == datetime.datetime(2020, 1, 6)):
        variable_list = [(31, ["prate"]), (8, ["t2m"]), (1, ["r"]) , (6, ["soilw"]), (7, ["u10", "v10"])]
    elif (date >= datetime.datetime(2019, 11, 8) and date <= datetime.datetime(2021, 3, 22)):
        variable_list = [(48, ["prate"]), (11, ["t2m"]), (20, ["r"]) , (6, ["soilw"]), (10, ["u10", "v10"])]
    elif (date >= datetime.datetime(2021, 3, 23) and date <= datetime.datetime(2023, 4, 10)):
        variable_list = [(49, ["prate"]), (13, ["t2m"]), (24, ["r"]) , (8, ["soilw"]), (12, ["u10", "v10"])]
    
    depthBelowLandLayer = 0.0
    
    min_lat = -25
    max_lat = -17
    
    min_lon = -52
    max_lon = -42
    
    # CRIA O DATAFRAME
    data = pd.DataFrame()
    for variable in variable_list:
        df = datasets[variable[0]].get(variable[1]).to_dataframe()
        print (variable, "1:", len(df))
        
        if (variable[1][0]=='soilw'):
            df = df.loc[depthBelowLandLayer, :, :]
        
        latitudes = df.index.get_level_values("latitude")
        longitudes = df.index.get_level_values("longitude")
        
        novas_longitudes = longitudes.map(lambda lon: (lon-360) if (lon>180) else lon)
        
        df['longitude'] = novas_longitudes
        df['latitude'] = latitudes
            
        lat_filter = (df['latitude']>= min_lat) & (df['latitude']<=max_lat)
        lon_filter = (df['longitude']>=min_lon) & (df['longitude']<=max_lon)
        
        df = df.loc[lat_filter & lon_filter]
        
        df = df.set_index(['latitude', 'longitude'])
        
        print (variable, "2:", len(df))
        
        if (data.empty):
            data = df
        else:
            data = data.merge(df, on=['latitude', 'longitude'])
     
    # IMPRIME NA TELA INFORMAÇÕES DO DATAFRAME
    data.info()
    
    # CRIA GRÁFICOS
    plot = 0
    if (plot==1):
        for variable in variable_list:
            for v in variable[1]:
                plt.title(file+": "+v)
                plt.scatter([x[1] for x in data.index], [x[0] for x in data.index], c=data[v])
                plt.show()
            
    # USA A MÁSCARA
    num = 3
    
    area = pd.read_csv("rastert_areas-i3.txt", delim_whitespace=True, skiprows=6, \
                     names=list(range(20)))
    
    area_filter = data["time"].rename("filter")
    for i in range(area.shape[0]+1):
        for j in range(area.shape[1]+1):
            # print(max_lat-0.5*i, min_lon+0.5*j)
            if (i!=area.shape[0] and j!=area.shape[1] and area.iloc[i, j]==num):
                area_filter[max_lat-0.5*i, min_lon+0.5*j] = True
            else:
                area_filter[max_lat-0.5*i, min_lon+0.5*j] = False
                
    data = data.loc[area_filter]
                
    # CRIA SÉRIE DIÁRIA
    for variable in variable_list:
        for v in variable[1]:
            f = open(dir_files+"serie_diaria_"+v+".csv", "a")
            f.write(str(date)+", "+str(data[v].mean())+"\n")
            f.close()