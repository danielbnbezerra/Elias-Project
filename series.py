import pandas as pd
#import numpy as np
#from statsmodels.tsa.tsatools import detrend
from darts import TimeSeries
from darts.metrics import mape,rmse
#from darts.utils.statistics import check_seasonality, plot_acf

class GetSeries:
    def __init__(self, series_file):
        self.timeseries = self.make_timeseries(series_file)
        self.train, self.valid = self.train_valid_split()

    def make_timeseries(self, series_file):
        df = pd.read_csv(series_file)
        timeseries = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
        return timeseries

    def train_valid_split(self):
        train, valid = self.timeseries.split_before(int(self.timeseries.n_timesteps * 0.80)) #Por padrão separando em 80% treino e 20% validação
        return train, valid

class MetricModels:
    def __init__(self, series, predictions):
        self.mape = {}
        self.rmse = {}
        self.generate_metrics(series,predictions)

    def generate_metrics(self, valid, predictions):

        models = list(predictions.keys())
        for model in models:
            if model == "LHC":
                self.mape["LHC"] = mape(valid, predictions["LHC"][:len(valid)]) #Passar a pred do LHC pra TimeSeries
                self.rmse["LHC"] = rmse(valid, predictions["LHC"][:len(valid)])
            if model == "NBEATS":
                self.mape["NBEATS"] = mape(valid, predictions["NBEATS"][:len(valid)])
                self.rmse["NBEATS"] = rmse(valid, predictions["NBEATS"][:len(valid)])
            if model == "NHiTS":
                self.mape["NHiTS"] = mape(valid, predictions["NHiTS"][:len(valid)])
                self.rmse["NHiTS"] = rmse(valid, predictions["NHiTS"][:len(valid)])

#Aqui vamos realizar o teste de hipótese no modelo especificado para diferentes métodos disponíveis.

def  dagostino_k_squared_test(residuals):
  """
  This tests whether a sample differs from a normal distribution. It is based on D’Agostino and Pearson’s test that combines skew and
  kurtosis to produce an omnibus test of normality. In Python, scipy.stats.normaltest is used to test this. It gives the statistic which
  is s^2 + k^2, where s is the z-score returned by skew test and k is the z-score returned by kurtosis test and p-value, i.e., 2-sided chi
  squared probability for the hypothesis test. Alpha value is 0.05. The null hypothesis is rejected for all the variables suggesting that all
  the variables are not normally distributed. As in graphical analysis, “Height” variable was looking normal, here “Height” variable too is
  showing p-value 0 suggesting the variable is not normally distributed.
  """

  alpha = 0.05
  residuals_List = list(value[0] for value in residuals.values())
  statistic, pvalue = normaltest(residuals_List)
  dagostino_pearson_mape = single_mape(alpha,pvalue)
  #D'Agostino-Pearson's K-Squared Test Result:
  result = f"{'Dados DN' if pvalue >= alpha else 'Dados DNN'}."
  return (dagostino_pearson_mape, False, result) if pvalue < alpha else (0,True, result)

def jarque_bera_test(residuals):
  """
  This tests whether the sample has the skewness and kurtosis matching with a normal distribution, i.e., skewness=0 and kurtosis =3.
  The null hypothesis is same as D’Agostino’s K-squared test. The test statistic is always nonnegative, and if it is far from zero then
  it shows the data do not have a normal distribution. This test only works for more than 2000 data samples.
  In Python, scipy.stats.jarque_bera is used for the test. Below we can see that all variables’ test statistics is far from zero and
  p-values are 0 suggesting that all variables are not normally distributed.
  """

  alpha = 0.05
  residuals_List = list(value[0] for value in residuals.values())
  statistic, pvalue = jarque_bera(residuals_List)
  jarque_bera_mape = single_mape(alpha,pvalue)
  #Jarque-Bera Test Result:
  result = f"{'Dados DN' if pvalue >= alpha else 'Dados DNN'}."
  return (jarque_bera_mape, False, result) if pvalue < alpha else (0,True, result)

def anderson_darling_test(residuals):
  """
  This tests if sample is coming from a particular distribution. The null hypothesis is that the sample is drawn from a population following
  a particular distribution. For the Anderson-Darling test, the critical values depend on the distribution it is being tested. The distribution
  it takes are normal, exponential, logistic, or Gumbel (Extreme Value Type I) distributions. If the test statistic is larger than the critical
  value then for the corresponding significance level, the null hypothesis (i.e., the data come from the chosen distribution) can be rejected.
  """

  residuals_List = list(value[0] for value in residuals.values())
  statistic = anderson(residuals_List,dist='norm')
  anderson_darling_mape = single_mape(statistic[1][2],statistic[0])
  #Anderson-Darling Test Result:
  result = f"{'Dados DN' if statistic[0] <= statistic[1][2] else f'Dados DNN'}."
  return (anderson_darling_mape, False, result) if statistic[0] > statistic[1][2] else (0,True, result)

def kolmogorov_smirnov_test(residuals):
  """
  This is a non-parametric test i.e., it has no assumption about the distribution of the data. Kolmogorov-Smirnov test is used to understand how
  well the distribution of sample data conforms to some theoretical distribution. In this, we compare between some theoretical cumulative
  distribution function, (Ft(x)), and a samples’ cumulative distribution function , (Fs(x)) where the sample is a random sample with unknown
  cumulative distribution function Fs(x).
  """

  alpha = 0.05
  residuals_List = list(value[0] for value in residuals.values())
  statistic, pvalue = kstest(residuals_List,'norm')
  kolmogorov_smirnov_mape = single_mape(alpha,pvalue)
  #Kolmogorov-Smirnov Test Result:
  result = f"{'Dados DN' if pvalue >= alpha else 'Dados DNN'}."
  return (kolmogorov_smirnov_mape, False, result) if pvalue < alpha else (0,True, result)

def lilliefors_test(residuals):
  """
  This is also a normality test that is based on the Kolmogorov–Smirnov test. This is specifically used to test the null hypothesis that the
  sample comes from a normally distributed population, when the null hypothesis does not specify which normal distribution, i.e., it does not
  specify the expected value and variance of the distribution.
  """

  alpha = 0.05
  residuals_List = list(value[0] for value in residuals.values())
  statistic, pvalue = lilliefors(residuals_List, dist='norm', pvalmethod='table')
  lilliefors_mape = single_mape(alpha,pvalue)
  #Lilliefors Test Result:
  result = f"{'Dados DN' if pvalue >= alpha else 'Dados DNN'}."
  return (lilliefors_mape, False, result) if pvalue < alpha else (0,True, result)

def shapiro_wilk_test(residuals):
  """
  This test is most popular to test the normality.
  """

  alpha = 0.05
  residuals_List = list(value[0] for value in residuals.values())
  statistic, pvalue = shapiro(residuals_List)
  shapiro_wilk_mape = single_mape(alpha,pvalue)
  #Shapiro-Wilk Test Result:
  result = f"{'Dados DN' if pvalue >= alpha else 'Dados DNN'}."
  return (shapiro_wilk_mape, False, result) if pvalue < alpha else (0,True, result)

def eval_tests(residuals):
  tests = [{'name':'D\'Agostino\'s K-Squared', 'method_results': dagostino_k_squared_test(residuals)},
           {'name':'Jarque-Bera', 'method_results': jarque_bera_test(residuals)},
           {'name':'Anderson-Darling', 'method_results': anderson_darling_test(residuals)},
           {'name':'Kolmogorov-Smirnov', 'method_results': kolmogorov_smirnov_test(residuals)},
           {'name':'Lilliefors', 'method_results': lilliefors_test(residuals)},
           {'name':'Shapiro-Wilk', 'method_results': shapiro_wilk_test(residuals)}
           ]

  for index in range(len(tests)):
    if not tests[index]['method_results'][1]:
      observation = f"Valor calculado no teste {tests[index]['name']} tem um erro percentual de {tests[index]['method_results'][0]:.2f}% comparado ao respectivo limite."
    else:
      observation = f"Limite alcançado no teste {tests[index]['name']}."
    tests[index]['observation'] = observation

  conclusion_check = sum(tup[1] for tup in (test.get('method_results') for test in tests))

  if conclusion_check > 3:
    conclusion = "Dados seguem distribuição Normal."
  elif conclusion_check == 3:
    conclusion =  "Inconclusivo. Cheque 'Observações' para atingir uma conclusão definitiva."
  else:
    conclusion =  "Dados não seguem distribuição Normal."

  return tests, conclusion

def eval_tests_n_models(models_residuals):

  writer = pd.ExcelWriter('Testes de Normalidade dos Modelos.xlsx',engine='xlsxwriter')
  workbook=writer.book
  writer.sheets['Resultados'] = workbook.add_worksheet('Resultados')

  model_names = list(model['name'] for model in models_residuals)
  column_names = ['D\'Agostino\'s K-Squared',
                  'Jarque-Bera',
                  'Anderson-Darling',
                  'Kolmogorov-Smirnov',
                  'Lilliefors',
                  'Shapiro-Wilk']

  for index in range(len(models_residuals)):
    data=[]
    observations=[]
    observation_values=[]
    model_tests, model_conclusion = eval_tests(models_residuals[index]['residual'])
    for test in model_tests:
      data.append(test['method_results'][2])
      observations.append(test['observation'])
      observation_values.append(f"{test['method_results'][0]:.2f}%")
    dfModelTests = pd.DataFrame([data,observation_values,'','','',''], index = [models_residuals[index]['name'],'','','','',''], columns = column_names)
    dfModelTests['Conclusão'] = [model_conclusion,'','','','','']
    dfModelTests['Observações'] = observations
    dfModelTests.to_excel(writer, sheet_name='Resultados',startrow=7*index, startcol=0)
  dfCaption = pd.DataFrame({'Legenda': ['DN','DNN'],
                            'Definição':['Distribuição Normal','Distribuição Não Normal']})
  dfCaption.set_index('Legenda', inplace=True)
  dfCaption.to_excel(writer, sheet_name='Resultados',startrow=len(models_residuals)*7 + 2, startcol=0)

  writer.save()
  writer.close()

#Função para gerar as colunas dos meses até o total de 4 anos de previsão.

def monthlist_Fast(dates):
  start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
  total_months = lambda dt: dt.month + 12 * dt.year
  mlist = []
  for tot_m in range(total_months(start)-1, total_months(end)):
    y, m = divmod(tot_m, 12)
    mlist.append(datetime(y, m+1, 1).strftime("%b/%Y"))
  return mlist