import pandas as pd

from darts import TimeSeries


class GetSeries:
    def __init__(self, series_files, train_percent):
        self.train_percent = train_percent
        self.prate=None
        self.prate_t_minus_3=None
        self.prate_t_minus_5=None
        self.flow=None
        self.make_timeseries(series_files)
        self.train_cov, self.valid_cov, self.train_target, self.valid_target= self.train_valid_split(self.train_percent)

    def make_timeseries(self, series_file):
        max_window = 5  # maior janela

        df_prate = pd.read_csv(series_file["prate"], header=None, names=["date", "prate"])
        df_prate["date"] = pd.to_datetime(df_prate["date"])

        df_flow = pd.read_csv(series_file["flow"], header=None, names=["date", "flow"])
        df_flow["date"] = pd.to_datetime(df_flow["date"])

        #Criar séries cumulativas
        df_prate_t_minus_3 = self.make_cumulative_series(df_prate, 3)
        df_prate_t_minus_5 = self.make_cumulative_series(df_prate, 5)

        #Cortar os primeiros 5 valores para todas as séries (tamanho da janela máxima)
        df_prate = df_prate.iloc[max_window - 1:]
        df_prate_t_minus_3 = df_prate_t_minus_3.iloc[max_window - 1:]
        df_prate_t_minus_5 = df_prate_t_minus_5.iloc[max_window - 1:]
        df_flow = df_flow.iloc[max_window - 1:]

        self.prate = TimeSeries.from_dataframe(df_prate, 'date', 'prate')
        self.prate_t_minus_3 = TimeSeries.from_dataframe(df_prate_t_minus_3, 'date', 'prate_t_minus_3')
        self.prate_t_minus_5 = TimeSeries.from_dataframe(df_prate_t_minus_5, 'date', 'prate_t_minus_5')
        self.prate_covariates = self.prate.stack(self.prate_t_minus_5).stack(self.prate_t_minus_3)
        self.flow = TimeSeries.from_dataframe(df_flow, 'date', 'flow')

    def make_cumulative_series(self, df, days):
        return pd.DataFrame({
            "date": df["date"],
            f"prate_t_minus_{days}": df["prate"].rolling(window=days, min_periods=1).sum()
        })

    def train_valid_split(self, train_percent):

        train_cov, valid_cov = self.prate_covariates.split_before(int(self.prate_covariates.n_timesteps * train_percent))
        train_target, valid_target = self.flow.split_before(int(self.flow.n_timesteps * train_percent)) #Por padrão separando em 80% treino e 20% validação
        return train_cov, valid_cov, train_target, valid_target

    def update_date_interval(self, date_start, date_end):

        # Converte para Timestamp se necessário
        if not isinstance(date_start, pd.Timestamp):
            date_start = pd.to_datetime(date_start)
        if not isinstance(date_end, pd.Timestamp):
            date_end = pd.to_datetime(date_end)

        #Recorte das séries
        self.prate = self.prate.slice(date_start, date_end)
        self.prate_t_minus_3 = self.prate_t_minus_3.slice(date_start, date_end)
        self.prate_t_minus_5 = self.prate_t_minus_5.slice(date_start, date_end)
        self.flow = self.flow.slice(date_start, date_end)

        # Atualiza covariates
        self.prate_covariates = self.prate.stack(self.prate_t_minus_5).stack(self.prate_t_minus_3)

        # Atualiza treino e validação, se já tiver sido definido
        if hasattr(self, "train_cov") and hasattr(self, "train_target"):
            self.train_cov, self.valid_cov, self.train_target, self.valid_target = self.train_valid_split(self.train_percent)

# #Funções de Aquisição de Dados
#
# import os
# import subprocess
# import tarfile
# import cfgrib
#
# import matplotlib.pyplot as plt
#
# from datetime import timedelta
#
#
# def download_GFS_data(orderID="HAS012640249"): #ID do request atual é o padrão. Idealmente não existe padrão pois os request duram dias.
#
#     # DEFINIR DIRETÓRIO DOS ARQUIVOS .g2.tar
#     dir_arq = "DadosGFS\\raw"
#     os.makedirs(dir_arq, exist_ok=True)  # <- garante que a pasta existe
#
#     # DEFINIR DATAS
#     dia_inicio, mes_inicio, ano_inicio = 10, 4, 2014
#     dia_fim, mes_fim, ano_fim = 9, 4, ano_inicio + 1
#
#     dias = [f"{x:02d}" for x in range(1, 32)]
#     meses = [f"{x:02d}" for x in range(1, 13)]
#     anos = [str(x) for x in range(ano_inicio, ano_fim + 1)]
#
#     # Construir o script PowerShell como string
#     ps_script = f"$baseUri = 'https://www.ncei.noaa.gov/pub/has/model/{orderID}'\n$files = @(\n"
#
#     for ano in anos:
#         for mes in meses:
#             for dia in dias:
#                 include = False
#                 if int(ano) == ano_inicio and (
#                         int(mes) > mes_inicio or (int(mes) == mes_inicio and int(dia) >= dia_inicio)):
#                     include = True
#                 elif int(ano) == ano_fim and (int(mes) < mes_fim or (int(mes) == mes_fim and int(dia) <= dia_fim)):
#                     include = True
#                 elif int(ano) != ano_inicio and int(ano) != ano_fim:
#                     include = True
#
#                 if include:
#                     out_file = f"{dir_arq}\\gfsanl_4_{ano}{mes}{dia}00.g2.tar"
#                     if (int(mes) == mes_fim and int(dia) == dia_fim):
#                         ps_script += f"    @{{ Uri = \"$baseUri/gfsanl_4_{ano}{mes}{dia}00.g2.tar\"; OutFile = \"{out_file}\" }}\n"
#                     else:
#                         ps_script += f"    @{{ Uri = \"$baseUri/gfsanl_4_{ano}{mes}{dia}00.g2.tar\"; OutFile = \"{out_file}\" }},\n"
#
#     ps_script += ")\n\n$jobs = @()\nforeach ($file in $files) {\n"
#     ps_script += "    $jobs += Start-ThreadJob -Name $file.OutFile -ScriptBlock { Invoke-WebRequest @using:file }\n}\n"
#     ps_script += "Write-Host 'Downloads started...'\nWait-Job -Job $jobs\nforeach ($job in $jobs) { Receive-Job -Job $job }\n"
#
#     ps_filename = "baixar_gfs.ps1"
#     with open(ps_filename, "w", encoding="utf-8") as f:
#         f.write(ps_script)
#
#     # Executar diretamente no PowerShell 7
#     pwsh_path = r"C:\Program Files\PowerShell\7\pwsh.exe"
#     subprocess.run([pwsh_path, "-File", ps_filename])
#
#     if os.path.exists(ps_filename):
#         os.remove(ps_filename) #
#
# def extract_from_raw_data(dir_arq="DadosGFS"):
#
#     # Diretório onde estão os arquivos .g2.tar
#
#     # Lista arquivos .tar dentro da pasta
#     arq_lista = [arq for arq in os.listdir(dir_arq) if arq.endswith(".tar")]
#
#     # Extrai apenas o arquivo *_0000_003.grb2 de cada .tar
#     for arq in arq_lista:
#         caminho = os.path.join(dir_arq, arq)
#         try:
#             with tarfile.open(caminho, "r") as tar:
#                 # monta o nome esperado do arquivo
#                 arq_split = arq.split('_')
#                 nome = arq_split[0][:3] + "_" + arq_split[1] + "_" + arq_split[2][:-9] + "_0000_003.grb2"
#
#                 # procura dentro do tar
#                 membro = next((m for m in tar.getmembers() if m.name.endswith(nome)), None)
#
#                 if membro:
#                     print(f"Extraindo {membro.name} de {arq}...")
#                     tar.extract(membro, path=f"{dir_arq}\\extraidos\\")
#                 else:
#                     print(f"[AVISO] Arquivo {nome} não encontrado dentro de {arq}")
#
#         except Exception as e:
#             print(f"[ERRO] Não foi possível abrir {arq}: {e}")
#
# def code_grib2():
#     # DEFINIR DIRETÓRIO DOS ARQUIVOS .grb2
#     dir_files = "DadosGFS//"
#
#     # CRIA LISTA COM O NOME DOS ARQUIVOS
#     file_list = []
#     for file in os.listdir(dir_files):
#         if (file.split('.')[-1] == 'grb2'):
#             file_list.append(file)
#
#     for file in file_list:
#         datasets = cfgrib.open_datasets(dir_files + file)
#
#         # IMPRIME NA TELA AS VARIÁVEIS DO CONJUNTO DE DADOS
#         variable_dict = {}
#         dataset_number = 0
#         for dataset in datasets:
#             variable_dict[dataset_number] = []
#             for variable in dataset:
#                 variable_dict[dataset_number].append(
#                     (variable, dataset[variable].attrs["long_name"], dataset[variable].attrs["units"]))
#             dataset_number += 1
#
#         for dataset_number in variable_dict.keys():
#             print(dataset_number, "DIMENSION NAMES", str(list(datasets[dataset_number].dims)))
#             for variable in variable_dict[dataset_number]:
#                 print(dataset_number, variable)
#
#         # DEFINIR VARIÁVEIS, PRESSÃO, LATITUDE E LONGITUDE
#
#         date = file.split("_")[2]
#         date = datetime.datetime(int(str(date)[:4]), int(str(date)[4:6]), int(str(date)[6:]))
#
#         if (date >= datetime.datetime(2013, 1, 1) and date <= datetime.datetime(2015, 1, 14)):
#             variable_list = [(41, ["prate"]), (9, ["t2m"]), (17, ["r"]), (5, ["soilw"]), (8, ["u10", "v10"])]
#         elif (date >= datetime.datetime(2015, 1, 15) and date <= datetime.datetime(2016, 5, 11)):
#             variable_list = [(40, ["prate"]), (10, ["t2m"]), (18, ["r"]), (6, ["soilw"]), (9, ["u10", "v10"])]
#         elif (date >= datetime.datetime(2016, 5, 12) and date <= datetime.datetime(2019, 6, 12)):
#             variable_list = [(42, ["prate"]), (10, ["t2m"]), (18, ["r"]), (6, ["soilw"]), (9, ["u10", "v10"])]
#         elif (date >= datetime.datetime(2019, 6, 13) and date <= datetime.datetime(2019, 11, 7)):
#             variable_list = [(47, ["prate"]), (11, ["t2m"]), (19, ["r"]), (6, ["soilw"]), (10, ["u10", "v10"])]
#         elif (date == datetime.datetime(2020, 1, 6)):
#             variable_list = [(31, ["prate"]), (8, ["t2m"]), (1, ["r"]), (6, ["soilw"]), (7, ["u10", "v10"])]
#         elif (date >= datetime.datetime(2019, 11, 8) and date <= datetime.datetime(2021, 3, 22)):
#             variable_list = [(48, ["prate"]), (11, ["t2m"]), (20, ["r"]), (6, ["soilw"]), (10, ["u10", "v10"])]
#         elif (date >= datetime.datetime(2021, 3, 23) and date <= datetime.datetime(2023, 4, 10)):
#             variable_list = [(49, ["prate"]), (13, ["t2m"]), (24, ["r"]), (8, ["soilw"]), (12, ["u10", "v10"])]
#
#         depthBelowLandLayer = 0.0
#
#         min_lat = -25
#         max_lat = -17
#
#         min_lon = -52
#         max_lon = -42
#
#         # CRIA O DATAFRAME
#         data = pd.DataFrame()
#         for variable in variable_list:
#             df = datasets[variable[0]].get(variable[1]).to_dataframe()
#             print(variable, "1:", len(df))
#
#             if (variable[1][0] == 'soilw'):
#                 df = df.loc[depthBelowLandLayer, :, :]
#
#             latitudes = df.index.get_level_values("latitude")
#             longitudes = df.index.get_level_values("longitude")
#
#             novas_longitudes = longitudes.map(lambda lon: (lon - 360) if (lon > 180) else lon)
#
#             df['longitude'] = novas_longitudes
#             df['latitude'] = latitudes
#
#             lat_filter = (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)
#             lon_filter = (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
#
#             df = df.loc[lat_filter & lon_filter]
#
#             df = df.set_index(['latitude', 'longitude'])
#
#             print(variable, "2:", len(df))
#
#             if (data.empty):
#                 data = df
#             else:
#                 data = data.merge(df, on=['latitude', 'longitude'])
#
#         # IMPRIME NA TELA INFORMAÇÕES DO DATAFRAME
#         data.info()
#
#         # CRIA GRÁFICOS
#         plot = 0
#         if (plot == 1):
#             for variable in variable_list:
#                 for v in variable[1]:
#                     plt.title(file + ": " + v)
#                     plt.scatter([x[1] for x in data.index], [x[0] for x in data.index], c=data[v])
#                     plt.show()
#
#         # USA A MÁSCARA
#         num = 3
#
#         area = pd.read_csv("Old/rastert_areas-i3.txt", delim_whitespace=True, skiprows=6, \
#                            names=list(range(20)))
#
#         area_filter = data["time"].rename("filter")
#         for i in range(area.shape[0] + 1):
#             for j in range(area.shape[1] + 1):
#                 # print(max_lat-0.5*i, min_lon+0.5*j)
#                 if (i != area.shape[0] and j != area.shape[1] and area.iloc[i, j] == num):
#                     area_filter[max_lat - 0.5 * i, min_lon + 0.5 * j] = True
#                 else:
#                     area_filter[max_lat - 0.5 * i, min_lon + 0.5 * j] = False
#
#         data = data.loc[area_filter]
#
#         # CRIA SÉRIE DIÁRIA
#         for variable in variable_list:
#             for v in variable[1]:
#                 f = open(dir_files + "serie_diaria_" + v + ".csv", "a")
#                 f.write(str(date) + ", " + str(data[v].mean()) + "\n")
#                 f.close()
#
# def code_variable_list():
#     # DEFINIR DIRETÓRIO DOS ARQUIVOS .grb2
#     dir_files = "DadosGFS//"
#
#     # CRIA LISTA COM O NOME DOS ARQUIVOS
#     file_list = []
#     for file in os.listdir(dir_files):
#         if (file.split('.')[-1] == 'grb2'):
#             file_list.append(file)
#
#     for file in file_list:
#         datasets = cfgrib.open_datasets(dir_files + file)
#
#         # IMPRIME NA TELA AS VARIÁVEIS DO CONJUNTO DE DADOS
#         variable_dict = {}
#         dataset_number = 0
#         for dataset in datasets:
#             variable_dict[dataset_number] = []
#             for variable in dataset:
#                 variable_dict[dataset_number].append(
#                     (variable, dataset[variable].attrs["long_name"], dataset[variable].attrs["units"]))
#             dataset_number += 1
#
#         f = open("lista_variaveis_gfs//" + file + "_variaveis.txt", "a")
#         f.write(file + "\n")
#         for dataset_number in variable_dict.keys():
#             f.write(str(dataset_number) + " DIMENSION NAMES " + str(list(datasets[dataset_number].dims)) + "\n")
#             for variable in variable_dict[dataset_number]:
#                 f.write(str(dataset_number) + " " + str(variable) + "\n")
#
#         f.close()
#
# def missing_dates_grb2():
#
#     # DEFINIR DIRETÓRIO DOS ARQUIVOS .g2.tar
#     dir_arq = "F:\\DadosGFS.grb2"
#
#     # CRIA LISTA COM AS DATAS DOS ARQUIVOS
#     data_lista = []
#     for arq in os.listdir(dir_arq):
#         if (arq.split('.')[-1] == 'grb2'):
#             data_arq = arq.split("_")[2]
#             data_arq = datetime(int(str(data_arq)[:4]), int(str(data_arq)[4:6]), int(str(data_arq)[6:8]))
#             data_lista.append(data_arq)
#
#     # CRIA LISTA COM DATAS FALTANTES
#     datas_faltando = []
#     data_anterior = None
#     for data_arq in data_lista:
#         if (data_anterior == None):
#             data_anterior = data_arq
#         else:
#             if (data_arq != data_anterior + timedelta(days=1)):
#                 data = data_anterior + timedelta(days=1)
#                 while (data != data_arq):
#                     datas_faltando.append(data)
#                     data += timedelta(days=1)
#             data_anterior = data_arq
#
#     f = open("datas_faltantes_grb2.txt", "a")
#     for data_arq in datas_faltando:
#         f.write(str(data_arq) + "\n")
#     f.close()
#
# def missing_dates_tar():
#     # DEFINIR DIRETÓRIO DOS ARQUIVOS .g2.tar
#     dir_arq = "F:\\DadosGFS.tar"
#
#     # CRIA LISTA COM AS DATAS DOS ARQUIVOS
#     data_lista = []
#     for arq in os.listdir(dir_arq):
#         if (arq.split('.')[-1] == 'tar'):
#             data_arq = arq.split("_")[2]
#             data_arq = datetime(int(str(data_arq)[:4]), int(str(data_arq)[4:6]), int(str(data_arq)[6:8]))
#             data_lista.append(data_arq)
#
#     # CRIA LISTA COM DATAS FALTANTES
#     datas_faltando = []
#     data_anterior = None
#     for data_arq in data_lista:
#         if (data_anterior == None):
#             data_anterior = data_arq
#         else:
#             if (data_arq != data_anterior + timedelta(days=1)):
#                 data = data_anterior + timedelta(days=1)
#                 while (data != data_arq):
#                     datas_faltando.append(data)
#                     data += timedelta(days=1)
#             data_anterior = data_arq
#
#     f = open("datas_faltantes_tar.txt", "a")
#     for data_arq in datas_faltando:
#         f.write(str(data_arq) + "\n")
#     f.close()
#
# def code_lstm():
#     from keras.models import Sequential
#     from keras.layers import LSTM, Dense
#
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.model_selection import KFold
#     from sklearn.metrics import mean_squared_error, mean_absolute_error
#     from datetime import datetime, timedelta
#     import numpy as np
#     import pandas as pd
#
#     # DEFINIR VARIÁVEIS
#
#     variaveis_series = ["prate", "t2m", "r", "soilw", "u10", "v10", "vazao"]
#
#     # CRIA O DATAFRAME
#
#     df = pd.DataFrame()
#     for v in variaveis_series:
#         if (df.empty):
#             df = pd.read_csv("DadosGFS\\serie_diaria_" + v + ".csv", names=["Data", v])
#         else:
#             df = df.merge(pd.read_csv("DadosGFS\\serie_diaria_" + v + ".csv", names=["Data", v]), on="Data")
#
#     numero_variaveis = len(df.columns) - 2
#
#     # DEFINIR O NÚMERO DE PASSOS NO TEMPO
#
#     passos_no_tempo = 365
#
#     # VALIDAÇÃO CRUZADA
#
#     lista_mse = []
#     lista_rmse = []
#     lista_mae = []
#     lista2_mse = []
#     lista2_rmse = []
#     lista2_mae = []
#     lista_vazao_pred = []
#     for i_train, i_test in KFold(5).split(df):
#         df_train = df.iloc[i_train]
#         df_test = df.iloc[i_test]
#
#         scaler = StandardScaler()
#         df_train.iloc[:, 1:] = scaler.fit_transform(df_train.iloc[:, 1:])
#         df_test.iloc[:, 1:] = scaler.transform(df_test.iloc[:, 1:])
#
#         # TRANSFORMA O DATAFRAME EM SEQUÊNCIAS
#
#         seq_train = []
#         for i in range(len(df_train) - passos_no_tempo + 1):
#             seq_train.append(df_train[i:i + passos_no_tempo])
#             print('1', i)
#         seq_test = []
#         for i in range(len(df_test) - passos_no_tempo + 1):
#             print('2', i)
#             seq_test.append(df_test[i:i + passos_no_tempo])
#
#         # SEPARA A DATA
#
#         seq_train_data = [x["Data"] for x in seq_train]
#         seq_test_data = [x["Data"] for x in seq_test]
#         seq_train = np.array([x.drop("Data", axis=1) for x in seq_train])
#         seq_test = np.array([x.drop("Data", axis=1) for x in seq_test])
#
#         # REMOVE SEQUÊNCIAS DESCONTÍNUAS DEVIDO ÀS DATAS FALTANTES
#
#         datas_faltantes = pd.read_csv("DadosGFS\\datas_faltantes_grb2.txt", names=["Data"])
#
#         lista_continua = []
#         for i in range(len(seq_train_data)):
#             continua = True
#             for j in range(len(datas_faltantes)):
#                 data_faltante = datas_faltantes.iloc[j][0]
#                 data = datetime(int(data_faltante[:4]), int(data_faltante[5:7]), int(data_faltante[8:10])) - timedelta(
#                     days=1)
#                 if (str(data) in list(seq_train_data[i].values) and str(data) != list(seq_train_data[i].values)[-1]):
#                     continua = False
#             if (continua == True):
#                 lista_continua.append(i)
#         seq_train = seq_train[lista_continua]
#         seq_train_data = list(seq_train_data[i] for i in lista_continua)
#         print('1')
#         lista_continua = []
#         for i in range(len(seq_test_data)):
#             continua = True
#             for j in range(len(datas_faltantes)):
#                 data_faltante = datas_faltantes.iloc[j][0]
#                 data = datetime(int(data_faltante[:4]), int(data_faltante[5:7]), int(data_faltante[8:10])) - timedelta(
#                     days=1)
#                 if (str(data) in list(seq_test_data[i].values) and str(data) != list(seq_test_data[i].values)[-1]):
#                     continua = False
#             if (continua == True):
#                 lista_continua.append(i)
#         seq_test = seq_test[lista_continua]
#         seq_test_data = list(seq_test_data[i] for i in lista_continua)
#         print('2')
#
#         # TREINA E TESTA O MODELO
#
#         model = Sequential()
#         model.add(LSTM(128, input_shape=(None, numero_variaveis)))
#         model.add(Dense(16))
#         model.add(Dense(1))
#         model.compile(loss='mse', optimizer='adam')
#
#         model.fit(seq_train[:, :, :-1], seq_train[:, -1, -1], epochs=10)
#
#         vazao_pred = model.predict(seq_test[:, :, :-1])
#
#         # AVALIA O ERRO DA VAZÃO COM PADRONIZAÇÃO
#
#         lista_mse.append(mean_squared_error(seq_test[:, -1, -1], vazao_pred))
#         lista_rmse.append(mean_squared_error(seq_test[:, -1, -1], vazao_pred, squared=False))
#         lista_mae.append(mean_absolute_error(seq_test[:, -1, -1], vazao_pred))
#
#         # AVALIA O ERRO DA VAZÃO SEM PADRONIZAÇÃO
#
#         vazao_pred = scaler.inverse_transform(
#             np.hstack([vazao_pred, vazao_pred, vazao_pred, vazao_pred, vazao_pred, vazao_pred, vazao_pred]))
#         vazao_pred = vazao_pred[:, -1]
#
#         vazao_test = seq_test[:, -1, -1].reshape([seq_test.shape[0], 1])
#         vazao_test = scaler.inverse_transform(
#             np.hstack([vazao_test, vazao_test, vazao_test, vazao_test, vazao_test, vazao_test, vazao_test]))
#         vazao_test = vazao_test[:, -1]
#
#         lista2_mse.append(mean_squared_error(vazao_test, vazao_pred))
#         lista2_rmse.append(mean_squared_error(vazao_test, vazao_pred, squared=False))
#         lista2_mae.append(mean_absolute_error(vazao_test, vazao_pred))
#
#         # VAZÃO ESTIMADA COM DATA
#
#         vazao_pred = pd.concat([pd.DataFrame([x.iloc[-1] for x in seq_test_data], columns=["Data"]), \
#                                 pd.DataFrame(vazao_pred, columns=["Vazao"])], \
#                                axis=1)
#         lista_vazao_pred.append(vazao_pred)
#
#     f = open("DadosGFS\\desempenho_lstm.txt", "a")
#     f.write("VAZÃO COM PADRONIZAÇÃO" + "\n")
#     f.write("MSE: " + str(lista_mse) + "\n")
#     f.write(str(pd.DataFrame(lista_mse).describe()) + "\n")
#     f.write("RMSE: " + str(lista_rmse) + "\n")
#     f.write(str(pd.DataFrame(lista_rmse).describe()) + "\n")
#     f.write("MAE: " + str(lista_mae) + "\n")
#     f.write(str(pd.DataFrame(lista_mae).describe()) + "\n")
#     f.write("VAZÃO SEM PADRONIZAÇÃO" + "\n")
#     f.write("MSE: " + str(lista2_mse) + "\n")
#     f.write(str(pd.DataFrame(lista2_mse).describe()) + "\n")
#     f.write("RMSE: " + str(lista2_rmse) + "\n")
#     f.write(str(pd.DataFrame(lista2_rmse).describe()) + "\n")
#     f.write("MAE: " + str(lista2_mae) + "\n")
#     f.write(str(pd.DataFrame(lista2_mae).describe()) + "\n")
#     f.close()
#
#     for i in range(len(lista_vazao_pred)):
#         lista_vazao_pred[i].to_csv("DadosGFS\\vazao_pred_" + str(i) + ".csv")