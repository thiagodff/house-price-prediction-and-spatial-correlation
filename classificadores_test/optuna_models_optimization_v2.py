import pandas as pd
import numpy as np
import optuna
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf


from pathlib import Path

# divide os dados em variáveis dependentes de rótulo (y) e variáveis independentes (x)
def split_data(data):
    y = data['preco']
    x = data.drop(columns='preco')
    y_index = y.index
    y = LabelEncoder().fit_transform(y)
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


def get_ranked_features_splits(data, method='pearson'):
    features_rank = {}

    for i, data_it in data.items():
        x_train = data_it['x_train']
        y_train = data_it['y_train']

        if method=='ANOVA':
            anova_values, _ = f_classif(x_train, y_train)
            feature_names = x_train.columns
            anova_ranking = sorted(zip(anova_values, feature_names), key = lambda x: x[0], reverse=True)
            anova_ranking = [x[1] for x in anova_ranking]
            features_rank[i] = anova_ranking

        elif method=='InfoGain':
            infogain_values = mutual_info_classif(x_train, y_train)
            feature_names = x_train.columns
            infogain_ranking = sorted(zip(infogain_values, feature_names), key=lambda x: x[1], reverse=True)
            infogain_ranking = [x[1] for x in infogain_ranking]
            features_rank[i] = infogain_ranking

        elif method=='OneR':
            features_file = f'weka/OneR_train{i}.json'
            with open(features_file) as json_file:
                oner_ranking = json.load(json_file)
            features_rank[i] = oner_ranking

        elif method in ['pearson', 'spearman', 'kendall']:
            corr_ranking = x_train.corrwith(y_train, method = method)
            corr_ranking = corr_ranking.abs()
            corr_ranking.sort_values(ascending = False, inplace = True)
            corr_ranking = corr_ranking.index
            features_rank[i] = corr_ranking

    return features_rank


def get_classifier(classifier_name, trial):
    # Setup values for the hyperparameters:

    if classifier_name == 'LogReg':
        logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e2, log=True)
        penalty = trial.suggest_categorical("penalty", 
                                               ["l1", "l2", "elasticnet"]
                                            )

        l1_ratio = None
        solver = 'saga'

        if penalty == 'l2':
            solver = 'lbfgs'
        elif penalty == 'elasticnet':
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            
        multi_class = trial.suggest_categorical("multi_class", 
                                               ["ovr", "multinomial"]
                                            )
    
        return LogisticRegression(
            C=logreg_c, penalty=penalty,
            solver=solver, multi_class=multi_class,
            l1_ratio=l1_ratio)

    elif classifier_name == 'RandomForest':
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)

        criterion = trial.suggest_categorical("criterion", 
                                               ["gini", "entropy"]
                                            )


        max_features = trial.suggest_categorical("max_features", 
                                               ["sqrt", "log2"]
                                            )

        min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 1.0)

        return RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators, 
            criterion=criterion, max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
        )

    elif classifier_name == 'SVM':
        kernel = trial.suggest_categorical("kernel",
                                          ['rbf', 'poly'])
        
        svm_c = trial.suggest_float("svm_c", 0.1, 2.0)

        decision_function_shape = trial.suggest_categorical("decision_function_shape",
                                          ['ovo', 'ovr'])

        shrinking = trial.suggest_categorical("shrinking",
                                          [True, False])


        coef0 = trial.suggest_float("coef0", 0.0, 1.0)

        print(f"{kernel=}, {svm_c=}, {decision_function_shape=}, \
            {shrinking=}, {coef0=} "
            )
        
        return SVC(
            kernel=kernel, C=svm_c,
            decision_function_shape=decision_function_shape,
            shrinking=shrinking, coef0=coef0
            )

    elif classifier_name == 'KNN':
        leaf_size = trial.suggest_int("leaf_size", 10, 40, log=True)
        n_neighbors = trial.suggest_int("n_neighbors", 3, 7, log=True)

        weights = trial.suggest_categorical("weights",
                                          ['uniform', 'distance'])

        p = trial.suggest_categorical("p",
                                          [1, 2])

        return KNeighborsClassifier(
            n_neighbors=n_neighbors, leaf_size=leaf_size,
            weights=weights, p=p)

    elif classifier_name == 'XGB':
        params = {
            'verbosity': 0,
            'objective': 'multi:softmax',
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'min_child_weight': trial.suggest_float("min_child_weight", 0.5, 5.0, log=True),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
            'use_label_encoder':False
        }
        
        return xgb.XGBClassifier(**params)

    if classifier_name == 'MLP':
        hidden_layers = trial.suggest_categorical("hidden_layers",
                                          [1, 2, 3])
                                          
        neurons = trial.suggest_categorical("neurons",
                                          [16, 32, 64, 128, 256]
                                          )

        learning_rate_type = trial.suggest_categorical("learning_rate_type",
                                          ['constant', 'invscaling', 'adaptive'])

        # optimizer = trial.suggest_categorical("optimizer",
                                        #   [ "adam"])
        optimizer = "adam"

        learning_rate_init = 0.001
        if optimizer=="adam":
            learning_rate_init = trial.suggest_categorical("learning_rate_init",
                                          [0.0001, 0.001, 0.01])

        hidden_layer_sizes = (neurons,)
        if hidden_layers==2:
            hidden_layer_sizes = (neurons,neurons,)
        if hidden_layers==3:
            hidden_layer_sizes = (neurons,neurons,neurons,)

        mlp = MLPClassifier(
            hidden_layer_sizes = hidden_layer_sizes, 
            solver = optimizer, 
            learning_rate = learning_rate_type,
            learning_rate_init = learning_rate_init
        )

        return mlp

    elif classifier_name=='NeuralNet':
        hidden_layers = trial.suggest_categorical("hidden_layers",
                                          [1, 2, 3])
                                          
        neurons = trial.suggest_categorical("neurons",
                                          [4, 8, 16, 32, 64, 128, 256]
                                          )


        learning_rate = trial.suggest_categorical("learning_rate_init",
                                          [0.001, 0.003, 0.01, 0.03])

        layers = []

        for _ in range(hidden_layers):
            layers.append(
                tf.keras.layers.Dense(neurons, activation='relu')
            )

        layers.append(
            tf.keras.layers.Dropout(dropout_value)
        )

        layers.append(
            tf.keras.layers.Dense(81, activation='softmax')
        )

        neuralnet = tf.keras.Sequential(layers)

        neuralnet.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        )
        
        return neuralnet

    else:
        raise ValueError('Classifier name not defined')


def report_features(feature_ranks, method):

    features_output = dict()
    for k,v in feature_ranks.items():
        features_output[k] = list(v)[0:10]

    with open(f'features_{method}.json', 'w', encoding='utf-8') as f:
        json.dump(features_output, f, ensure_ascii=False, indent=4)


def objective(trial):
    classifier = get_classifier(classifier_name, trial)
    score_list = []

    for i, data_it in data.items():

        # Slicing data
        x_train, x_test = data_it['x_train'], data_it['x_test']
        y_train, y_test = data_it['y_train'], data_it['y_test']

        # Filter features
        selected_features = feature_ranks[i]
        selected_features = selected_features[0:k]
        x_train = x_train.loc[:, selected_features]
        x_test = x_test.loc[:, selected_features]

        if classifier_name=="NeuralNet":
            y_train = tf.keras.utils.to_categorical(y_train, 81)
            y_test = tf.keras.utils.to_categorical(y_test, 81)
            classifier.fit(x_train, y_train, verbose=0)
            y_pred = classifier.predict(x_test)
            y_test = np.argmax(y_test, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

        else:
            #Fitting
            if classifier_name == 'XGB':
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)

        #Scoring
        score_it = accuracy_score(y_test, y_pred)
        score_list.append(score_it)

    avg_accuracy = sum(score_list) / len(score_list)

    return avg_accuracy


# Global variables
PATH = Path("optuna_studies")
df = pd.read_csv('dataset_imoveis_2.csv')
x, y = split_data(df)
indexes_list = get_split_indexes(x, y)
scalers = fit_scalers(x, indexes_list)
data = normalize_data(x, y, scalers, indexes_list)
dropout_value=0.5

# Tests parameters
methods = ["InfoGain", "ANOVA", "pearson", "spearman", "OneR"]
# classifiers = ["LogReg", "KNN", "XGB", "MLP"]
classifiers = ["NeuralNet"]
# LogReg e MLP warn, não convergiu no teste
# XGB error

k_list = [
    1, 4, 15, 30, 75, 500, 1100, 1700, 2548
]
# k_list = [
#     1, 2, 3, 4, 
#     5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 
#     300, 500, 700, 900, 1100, 1300, 1500, 
#     1700, 2000, 2300, 2548
# ]

n_trials = 50

# todos 2
# SVM 1, 3

for method in methods:
    feature_ranks = get_ranked_features_splits(data, method=method)
    #report_features(feature_ranks, method)

    path = PATH / method

    for k in k_list:
        for classifier_name in classifiers:

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)#timeout=10*60)

            study_df = study.trials_dataframe()
            study_df.sort_values(by='value', ascending=False, inplace=True)
            study_df.to_csv(path / f'{classifier_name}_k{k}.csv', index=False)
