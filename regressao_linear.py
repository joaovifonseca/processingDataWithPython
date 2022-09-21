# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 21:48:59 2022

@author: vitor
"""

#%% Pacotes
import pandas as pd

#%% Importando base
df = pd.read_csv('dataset/qualidade_vinhos.csv')

#%% Consultando a base dados
print(df.info())

#%% Limpando os dados
df.drop(df.columns[[12]], axis=1, inplace=True)

#%% Definindo variáveis X e y
X = df.iloc[:,0:11]
y = df.iloc[:,11]

print(f"Tamanho de X: {X.shape}")
print(f"Tamanho de y: {y.shape}")

#%% Selecionando base de treino e teste (25%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=123)

#%% Visualizando os tamanhos
print(f"Tamanho de X_train: {X_train.shape}")
print(f"Tamanho de X_test: {X_test.shape}")
print(f"Tamanho de y_train: {y_train.shape}")
print(f"Tamanho de y_test: {y_test.shape}")

#%% Rodando o modelo

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X_train, y_train)

clf.coef_.tolist()

#%% Coeficientes

# Coeficientes do modelo
for feature, coef in zip(X.columns.tolist(), clf.coef_.tolist()):
    print(f"{feature}: {round(coef, 2)}")

# Constante do modelo
print(f"Constante: {round(clf.intercept_, 2)}")

#%% R Quadrado

from sklearn.metrics import r2_score

y_pred = clf.predict(X_test)

print(f"R Quadrado de Teste: {r2_score(y_test, y_pred):1.1f}")


#%% Erro do Modelo

from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = clf.predict(X_test)

print(f"MSE de Teste: {mean_squared_error(y_test, y_pred):1.1f}")
print(f"MAE de Teste: {mean_absolute_error(y_test, y_pred):1.1f}")

#%% Fim