from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# DEFINIR VARIÁVEIS

variaveis_series = ["prate", "t2m", "r", "soilw", "u10", "v10", "vazao"]

# CRIA O DATAFRAME

df = pd.DataFrame()
for v in variaveis_series:
    if (df.empty):
        df = pd.read_csv("DadosGFS\\serie_diaria_"+v+".csv", names=["Data", v])
    else:
        df = df.merge(pd.read_csv("DadosGFS\\serie_diaria_"+v+".csv", names=["Data", v]), on="Data")

numero_variaveis = len(df.columns)-2

# DEFINIR O NÚMERO DE PASSOS NO TEMPO

passos_no_tempo = 365

# VALIDAÇÃO CRUZADA

lista_mse = []
lista_rmse = []
lista_mae = []
lista2_mse = []
lista2_rmse = []
lista2_mae = []
lista_vazao_pred = []
for i_train, i_test in KFold(5).split(df):
    df_train = df.iloc[i_train]
    df_test = df.iloc[i_test]

    scaler = StandardScaler()
    df_train.iloc[:, 1:] = scaler.fit_transform(df_train.iloc[:, 1:])
    df_test.iloc[:, 1:] = scaler.transform(df_test.iloc[:, 1:])

    # TRANSFORMA O DATAFRAME EM SEQUÊNCIAS
    
    seq_train = []
    for i in range(len(df_train)-passos_no_tempo+1):
        seq_train.append(df_train[i:i+passos_no_tempo])
        print('1', i)
    seq_test = []
    for i in range(len(df_test)-passos_no_tempo+1):
        print('2', i)
        seq_test.append(df_test[i:i+passos_no_tempo])
        
    # SEPARA A DATA
    
    seq_train_data = [x["Data"] for x in seq_train]
    seq_test_data = [x["Data"] for x in seq_test]
    seq_train = np.array([x.drop("Data", axis=1) for x in seq_train])
    seq_test = np.array([x.drop("Data", axis=1) for x in seq_test])

    # REMOVE SEQUÊNCIAS DESCONTÍNUAS DEVIDO ÀS DATAS FALTANTES
    
    datas_faltantes = pd.read_csv("DadosGFS\\datas_faltantes_grb2.txt",  names=["Data"])
    
    lista_continua = []
    for i in range(len(seq_train_data)):
        continua = True
        for j in range(len(datas_faltantes)):
            data_faltante = datas_faltantes.iloc[j][0]
            data = datetime(int(data_faltante[:4]), int(data_faltante[5:7]), int(data_faltante[8:10])) - timedelta(days=1)
            if (str(data) in list(seq_train_data[i].values) and str(data)!=list(seq_train_data[i].values)[-1]):
                continua = False
        if (continua==True):
            lista_continua.append(i)
    seq_train = seq_train[lista_continua]
    seq_train_data = list(seq_train_data[i] for i in lista_continua)
    print ('1')
    lista_continua = []
    for i in range(len(seq_test_data)):
        continua = True
        for j in range(len(datas_faltantes)):
            data_faltante = datas_faltantes.iloc[j][0]
            data = datetime(int(data_faltante[:4]), int(data_faltante[5:7]), int(data_faltante[8:10])) - timedelta(days=1)
            if (str(data) in list(seq_test_data[i].values) and str(data)!=list(seq_test_data[i].values)[-1]):
                continua = False
        if (continua==True):
            lista_continua.append(i)
    seq_test = seq_test[lista_continua]
    seq_test_data = list(seq_test_data[i] for i in lista_continua)
    print ('2')
    
    # TREINA E TESTA O MODELO
        
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, numero_variaveis)))
    model.add(Dense(16))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(seq_train[:, :, :-1], seq_train[:, -1, -1], epochs=10)
    
    vazao_pred = model.predict(seq_test[:, :, :-1])
    
    # AVALIA O ERRO DA VAZÃO COM PADRONIZAÇÃO
    
    lista_mse.append(mean_squared_error(seq_test[:, -1, -1], vazao_pred))
    lista_rmse.append(mean_squared_error(seq_test[:, -1, -1], vazao_pred, squared=False))
    lista_mae.append(mean_absolute_error(seq_test[:, -1, -1], vazao_pred))
    
    # AVALIA O ERRO DA VAZÃO SEM PADRONIZAÇÃO

    vazao_pred = scaler.inverse_transform(np.hstack([vazao_pred, vazao_pred, vazao_pred, vazao_pred, vazao_pred, vazao_pred, vazao_pred]))
    vazao_pred = vazao_pred[:, -1]
    
    vazao_test = seq_test[:, -1, -1].reshape([seq_test.shape[0], 1])
    vazao_test = scaler.inverse_transform(np.hstack([vazao_test, vazao_test, vazao_test, vazao_test, vazao_test, vazao_test, vazao_test]))
    vazao_test = vazao_test[:, -1]
    
    lista2_mse.append(mean_squared_error(vazao_test, vazao_pred))
    lista2_rmse.append(mean_squared_error(vazao_test, vazao_pred, squared=False))
    lista2_mae.append(mean_absolute_error(vazao_test, vazao_pred))
    
    # VAZÃO ESTIMADA COM DATA

    vazao_pred = pd.concat([pd.DataFrame([x.iloc[-1] for x in seq_test_data], columns=["Data"]),\
                                 pd.DataFrame(vazao_pred, columns=["Vazao"])],\
                                axis=1)
    lista_vazao_pred.append(vazao_pred)

f = open("DadosGFS\\desempenho_lstm.txt", "a")
f.write("VAZÃO COM PADRONIZAÇÃO"+"\n")
f.write("MSE: "+str(lista_mse)+"\n")
f.write(str(pd.DataFrame(lista_mse).describe())+"\n")
f.write("RMSE: "+str(lista_rmse)+"\n")
f.write(str(pd.DataFrame(lista_rmse).describe())+"\n")
f.write("MAE: "+str(lista_mae)+"\n")
f.write(str(pd.DataFrame(lista_mae).describe())+"\n")
f.write("VAZÃO SEM PADRONIZAÇÃO"+"\n")
f.write("MSE: "+str(lista2_mse)+"\n")
f.write(str(pd.DataFrame(lista2_mse).describe())+"\n")
f.write("RMSE: "+str(lista2_rmse)+"\n")
f.write(str(pd.DataFrame(lista2_rmse).describe())+"\n")
f.write("MAE: "+str(lista2_mae)+"\n")
f.write(str(pd.DataFrame(lista2_mae).describe())+"\n")
f.close()

for i in range(len(lista_vazao_pred)):
    lista_vazao_pred[i].to_csv("DadosGFS\\vazao_pred_"+str(i)+".csv")