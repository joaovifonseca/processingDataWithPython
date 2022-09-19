# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Carregando Pacotes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

#%% Carregando o Dataset
#fonte: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud?resource=download
df = pd.read_csv("card_transdata.csv")

#%% Visualizando o Dataset
print(df.info())

print(df.describe())

#%% Verificando a proporção de transações fraudulentas
fraude_count = df[df["fraud"] == 1]["fraud"].count()
n_fraude_count = df[df["fraud"] == 0]["fraud"].count()
print("Numero de transacoes fraudulentas:", fraude_count)
print("Numero de transacoes nao fraudulentas", n_fraude_count)
print("Percentual de fraudes:", fraude_count / (fraude_count + n_fraude_count) * 100)

#%% Visualização Gráfica
categories = ["Nao_Fraude", "Fraude"]
xpos = np.array([0, 1])
plt.xticks(xpos, categories)
plt.xlabel("Tipo Transacao")
plt.ylabel("Contagem Transacao")
plt.title("Transacoes por Tipo")
plt.bar(xpos[0], n_fraude_count, width= 0.7, color = "g")
plt.bar(xpos[1], fraude_count, width = 0.7, color="r")

#%% Preparando a Base  - Separando as variáveis X e Y
x = df.drop("fraud", axis = 1).values
y = df["fraud"].values

print(f"Tamanho de X: {x.shape}")
print(f"Tamanho de y: {y.shape}")
