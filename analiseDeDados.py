# -*- coding: utf-8 -*-

# Primeira parte da atividade - análise de dados e apresentação gráfica
# Utilizando um dataset de gastos de um orgão público para analisar qual área que está gastando mais e apresentar isso gráficamente

#%% Pacotes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Lista Cabeçalho
cabecalhos = [
    "CODIGO_ORGAO_SUPERIOR",
    "NOME_ORGAO_SUPERIOR",
    "CODIGO_ORGAO",
    "NOME_ORGAO",
    "CODIGO_UNIDADE_GESTORA",
    "NOME_UNIDADE_GESTORA",
    "ANO_EXTRATO",
    "MES_EXTRATO",
    "CPF_PORTADOR",
    "NOME_PORTADOR",
    "DOCUMENTO_FAVORECIDO",
    "NOME_FAVORECIDO",
    "TRANSACAO",
    "DATA_TRANSACAO",
    "VALOR_TRANSACAO"
]

#%% Importando base e trasnformando em dataframe
dr = pd.read_csv('dataset/202201_CPGF.csv', encoding="latin-1", sep=';')
df = pd.DataFrame(dr)

#%% Consultando a base dados
print(df.info())

#%%  Renomeando as variáveis
df.columns = cabecalhos

#%% Alterando a categoria da variável VALOR_TRANSACAO
df['VALOR_TRANSACAO'] = df['VALOR_TRANSACAO'].str.replace(',', '').astype(float)

#%% Consultando as alterações
print(df.info())

#%% Agrupando pelo valor da transação e sumarizando pela soma
df_filtro_val_trans = df.groupby('NOME_ORGAO')['VALOR_TRANSACAO'].sum()

print(df_filtro_val_trans)

#%% Visualizando graficamente
agrupamento_valor = df[['NOME_ORGAO', 'VALOR_TRANSACAO']].groupby(by='NOME_ORGAO').mean()

agrupamento_valor_ordenado = agrupamento_valor.sort_values(by=['VALOR_TRANSACAO'], ascending=False).reset_index()

sns.barplot(y="NOME_ORGAO", x="VALOR_TRANSACAO", data=agrupamento_valor_ordenado.head(10))
plt.title("Ranking de Valores por Órgãos")
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Órgãos', fontsize=12)
plt.show()

#%% Contagem de transações por Órgão
print(df['NOME_ORGAO_SUPERIOR'].value_counts())


#%% Gráfico de contagem

sns.countplot(y="NOME_ORGAO_SUPERIOR",
              data=df,
              order=df['NOME_ORGAO_SUPERIOR'].value_counts().index)

plt.title("Transações por Órgão")
plt.xlabel('Quantidade de Transações', fontsize=12)
plt.ylabel('Órgãos', fontsize=12)
plt.show()

#%%

# Segunda parte da atividade - Regressão Linear
# Realiza uma regressão da nota da qualidade do vinho baseado nas características de cada vinho

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

# Terceira parte - Regressão Logística
# Realiza o download do arquivo do kaggle pela API e utiliza o dataset para realizar a classificação de fraudes em cartões de crédito

#%% Carregando Pacotes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import opendatasets as opendatasets
from sklearn.metrics import accuracy_score

#%% Download e carregando o Dataset
opendatasets.download("https://www.kaggle.com/dhanushnarayananr/credit-card-fraud")
df = pd.read_csv("credit-card-fraud/card_transdata.csv")

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

X = df.iloc[:,0:7]
y = df.iloc[:,7]

#X = df.drop("fraud", axis = 1).values
#y = df["fraud"].values

print(f"Tamanho de X: {X.shape}")
print(f"Tamanho de y: {y.shape}")

#%% Quebrando o dataset em treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

#%% Visualizando
print('Tamanho de X_train: ', X_train.shape, '\n')
print('Tamanho de X_test: ', X_test.shape, '\n')
print('Tamanho de y_train: ', y_train.shape, '\n')
print('Tamanho de y_test: ', y_test.shape, '\n')

#%% Instanciando o modelo de regressão logística
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(n_jobs=-1, random_state=123)

#%% Fit do modelo

clf.fit(X_train, y_train)

#%% Obtendo os coeficientes do modelo
# Coeficientes do modelo
for feature, coef in zip(X.columns, clf.coef_[0].tolist()):
    print(f"{feature}: {round(coef, 3)}")

# Constante do modelo
print(f"Constante: {round(clf.intercept_[0], 3)}")

#%% Verificando a acurácia do modelo


y_train_true = y_train
y_train_pred = clf.predict(X_train)
y_test_true = y_test
y_test_pred = clf.predict(X_test)


print(f"Acurácia de Treino: {round(accuracy_score(y_train_true, y_train_pred), 2)}")
print('\n ---------------------------\n')
print(f"Acurácia de Teste: {round(accuracy_score(y_test_true, y_test_pred), 2)}")

#%% Fim