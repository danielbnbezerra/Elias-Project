import os

import numpy as np
import pandas as pd
import pymannkendall as mk

from scipy.stats import normaltest, shapiro, anderson
from darts.metrics import rmse, mae
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss

# RMSE - Root Mean Squared Error
def calc_rmse(actual, predicted):
    return rmse(actual, predicted)

# MAE - Mean Absolute Error usando
def calc_mae(actual, predicted):
    return mae(actual, predicted)

# NSE - Nash-Sutcliffe Efficiency
def nse(actual, predicted):
    actual = actual.values().flatten() if hasattr(actual, "values") else np.array(actual)
    predicted = predicted.values().flatten() if hasattr(predicted, "values") else np.array(predicted)
    numerator = np.sum((actual - predicted) ** 2)
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - numerator / denominator

# KGE - Kling-Gupta Efficiency
def kge(actual, predicted):
    actual = actual.values().flatten() if hasattr(actual, "values") else np.array(actual)
    predicted = predicted.values().flatten() if hasattr(predicted, "values") else np.array(predicted)

    r = np.corrcoef(actual, predicted)[0, 1]
    alpha = np.std(predicted) / np.std(actual)
    beta = np.mean(predicted) / np.mean(actual)
    result = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return result

# Mann-Kendall utilizando pymannkendall
def mann_kendall_test(series):
    result = mk.original_test(series)
    return {
        "trend": result.trend,
        "h": result.h,
        "p": result.p,
        "z": result.z,
        "tau": result.Tau,
        "slope": result.slope
    }

# Augmented Dickey-Fuller Test
def adf_test(series):
    alpha = 0.05
    result = adfuller(series)
    p_value = result[1]
    result_text = "Estacionária" if p_value < alpha else "Não Estacionária"
    return {
        "adf_statistic": result[0],
        "p_value": p_value,
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "result": result_text  # <- adicionar essa linha
    }

# KPSS Test
def kpss_test(series, regression='c'):
    alpha = 0.05
    result = kpss(series, regression=regression, nlags="auto")
    p_value = result[1]
    result_text = "Não Estacionária" if p_value < alpha else "Estacionária"
    return {
        "kpss_statistic": result[0],
        "p_value": p_value,
        "lags": result[2],
        "critical_values": result[3],
        "result": result_text  # adiciona o resultado interpretativo
    }

def dagostino_k_squared_test(residuals):
    """
        This tests whether a sample differs from a normal distribution. It is based on D’Agostino and Pearson’s test that combines skew and
        kurtosis to produce an omnibus test of normality. In Python, scipy.stats.normaltest is used to test this. It gives the statistic which
        is s^2 + k^2, where s is the z-score returned by skew test and k is the z-score returned by kurtosis test and p-value, i.e., 2-sided chi
        squared probability for the hypothesis test. Alpha value is 0.05. The null hypothesis is rejected for all the variables suggesting that all
        the variables are not normally distributed. As in graphical analysis, “Height” variable was looking normal, here “Height” variable too is
        showing p-value 0 suggesting the variable is not normally distributed.
        """
    alpha = 0.05
    residuals_list = list(value[0] for value in residuals.values())
    statistic, pvalue = normaltest(residuals_list)
    result_text = "Distribuição Normal" if pvalue >= alpha else "Distribuição Não Normal"
    return {"normal": pvalue >= alpha, "p_value": pvalue, "result": result_text}

def anderson_darling_test(residuals):
    """
        This tests if sample is coming from a particular distribution. The null hypothesis is that the sample is drawn from a population following
        a particular distribution. For the Anderson-Darling test, the critical values depend on the distribution it is being tested. The distribution
        it takes are normal, exponential, logistic, or Gumbel (Extreme Value Type I) distributions. If the test statistic is larger than the critical
        value then for the corresponding significance level, the null hypothesis (i.e., the data come from the chosen distribution) can be rejected.
        """

    residuals_list = list(value[0] for value in residuals.values())
    test_results = anderson(residuals_list, dist='norm')
    crit_val_5pct = test_results.critical_values[2]
    result_text = "Distribuição Normal" if test_results.statistic <= crit_val_5pct else "Distribuição Não Normal"
    return {
        "normal": test_results.statistic <= crit_val_5pct,
        "statistic": test_results.statistic,
        "critical_value_5pct": crit_val_5pct,
        "result": result_text
    }

def shapiro_wilk_test(residuals):
    """
        This test is most popular to test the normality.
        """

    alpha = 0.05
    residuals_list = list(value[0] for value in residuals.values())
    statistic, pvalue = shapiro(residuals_list)
    result_text = "Distribuição Normal" if pvalue >= alpha else "Distribuição Não Normal"
    return {"normal": pvalue >= alpha, "p_value": pvalue, "result": result_text}
