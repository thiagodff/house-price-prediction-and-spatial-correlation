import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from pathlib import Path

from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso

import optuna.visualization as vis
import matplotlib.pyplot as plt

# divide os dados em variáveis dependentes de rótulo (y) e variáveis independentes (x)
def split_data(data):
    y = data['preco']
    x = data.drop(columns='preco')
    y_index = y.index
    # y = LabelEncoder().fit_transform(y)
    y = pd.Series(data = y, index = y_index) # y é uma série indexada pelo índice original de y.

    return x, y

# divide o dataset de treinamento em 5 indices com treino e teste
def get_split_indexes(x, y):
    kf = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # kf = StratifiedKFold(n_splits=5)
    indexes = {}
    i=0
    for train_index, test_index in kf.split(x, y):
        indexes[i] = {
            'train': train_index,
            'test': test_index
        }
        i+=1

    return indexes

# normalizar os dados, garantindo que cada conjunto de treinamento tenha sua propria escala de valor
def fit_scalers(x, indexes):
    scalers = {}
    for i, idxs in indexes.items():
        x_train = x.loc[idxs['train'], :]
        scaler = MinMaxScaler()
        scaler = scaler.fit(x_train)
        scalers[i] = scaler

    return scalers


def normalize_data(x, y, scalers, indexes_list):
    data = {}

    for i, indexes in indexes_list.items():

        # Slicing data
        train_index = indexes['train']
        test_index = indexes['test']

        x_train, x_test = x.loc[train_index, :], x.loc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        # Normalization
        colunas = x_train.columns
        scaler = scalers[i]
        x_train = scaler.transform(x_train)
        x_train = pd.DataFrame(x_train, columns=colunas)
        x_test = scaler.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=colunas)

        data[i] = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }

    return data

def get_ranked_features_splits(data, method="RFE"):
    feature_ranks = {}

    for i, data_it in data.items():
        x_train, y_train = data_it['x_train'], data_it['y_train']

        if method == "RFE":
            estimator = Lasso(max_iter=7600)  # Utilizando Lasso como estimador no RFE
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(x_train, y_train)
            ranks = selector.ranking_
            feature_ranks[i] = x_train.columns[ranks == 1].tolist()

        elif method == "LASSO":
            estimator = Lasso(max_iter=7600)
            estimator.fit(x_train, y_train)
            ranks = np.abs(estimator.coef_)
            feature_ranks[i] = x_train.columns[np.argsort(ranks)][::-1][:n_features].tolist()

    return feature_ranks

# Definir a função para otimizar os hiperparâmetros do modelo
def objective(trial, model_name):
    if model_name == 'random_forest':
        # Definir os hiperparâmetros a serem otimizados para o Random Forest
        n_estimators = trial.suggest_int('n_estimators', 50, 1000, step=50)
        max_depth = trial.suggest_int('max_depth', 2, 30, step=2)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
    elif model_name == 'svr':
        # Definir os hiperparâmetros a serem otimizados para o svr
        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        degree = trial.suggest_int('degree', 2, 5)
        model = SVR(C=C, kernel=kernel, degree=degree)
        
    elif model_name == 'xgb':
        # Definir os hiperparâmetros a serem otimizados para o XGBoost
        n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        max_depth = trial.suggest_int('max_depth', 2, 30, step=2)
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

    elif model_name == 'gbm':
        gbm_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
        }
        model = GradientBoostingRegressor(**gbm_params)

    elif model_name == 'knn':
        # Definir os hiperparâmetros a serem otimizados para o KNN
        n_neighbors = trial.suggest_int('n_neighbors', 1, 10)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
    elif model_name == 'linear_regression':
        # Não há hiperparâmetros para otimizar na Regressão Linear
        model = LinearRegression()

    # Listas para armazenar as métricas (RMSE) de treinamento e teste
    rmse_train_scores = []
    rmse_test_scores = []
    
    # Loop sobre as divisões da validação cruzada
    #for train_index, test_index in shuffle_split.split(X):
    #    # Dividir o conjunto de dados em treinamento e teste
    #    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    ##    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    for i, data_it in data.items():

        # Slicing data
        x_train, x_test = data_it['x_train'], data_it['x_test']
        y_train, y_test = data_it['y_train'], data_it['y_test']

        # Filter features
        selected_features = feature_ranks[i]
        selected_features = selected_features[0:10]
        x_train = x_train.loc[:, selected_features]
        x_test = x_test.loc[:, selected_features]
        
        # Treinar o modelo com os dados de treinamento
        model.fit(x_train, y_train)
        
        # Fazer as previsões para os dados de treinamento e teste
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        
        # Calcular o RMSE para os dados de treinamento e teste
        rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Armazenar os RMSEs
        rmse_train_scores.append(rmse_train)
        rmse_test_scores.append(rmse_test)
    
    # Calcular a média dos RMSEs
    # mean_rmse_train = sum(rmse_train_scores) / len(rmse_train_scores)
    mean_rmse_test = sum(rmse_test_scores) / len(rmse_test_scores)
    
    # Retornar o RMSE médio do teste (usado pelo Optuna como objetivo de minimização)
    return mean_rmse_test

dataset = pd.read_csv('dataset_imoveis_new.csv')
# X = dataset.drop(columns='preco') # características (features)
# y = dataset['preco'] # preço (target)

x, y = split_data(dataset)
indexes_list = get_split_indexes(x, y)
scalers = fit_scalers(x, indexes_list)
data = normalize_data(x, y, scalers, indexes_list)

def formatNumberToCurrencyBRL(number):
    return "R$ {:,.2f}".format(number)

n_trials=80
PATH = Path("optuna_studies_regression")

# Criar um estudo Optuna

# Definir os modelos a serem otimizados
models = ['random_forest', 'svr', 'xgb', 'knn', 'gbm', 'linear_regression']

methods = ["RFE", "LASSO"]

n_features_list= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

for method in methods:
    for n_features in n_features_list:
        # Otimizar os hiperparâmetros para cada modelo
        feature_ranks = get_ranked_features_splits(data, method=method)
        print(feature_ranks)
        for model_name in models:
            n_jobs = 4 # Número de processos ou threads a serem usados
            study = optuna.create_study(direction='minimize')
            # study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials, n_jobs=n_jobs)

            print("Melhor RMSE:", study.best_value)

            trial_results = study.trials_dataframe()
            path = Path("optuna_studies_regression") / 'NO_SPATIAL' / method
            trial_results.to_csv(path / f'{n_features}_{model_name}.csv', index=False)

            # fig = optuna.visualization.plot_optimization_history(study)
            # fig.show()

            # Gráfico dos valores de cada hiperparâmetro
            # fig = optuna.visualization.plot_param_importances(study)
            # fig.show()
