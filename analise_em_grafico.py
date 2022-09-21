# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 17:09:00 2022

@author: vitor
"""
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
plt.xlabel('Valores',fontsize=12)
plt.ylabel('Órgãos',fontsize=12)
plt.show()

#%% Contagem de transações por Órgão
print(df['NOME_ORGAO_SUPERIOR'].value_counts())


#%% Gráfico de contagem

sns.countplot(y="NOME_ORGAO_SUPERIOR", 
              data=df,
              order = df['NOME_ORGAO_SUPERIOR'].value_counts().index)

plt.title("Transações por Órgão")
plt.xlabel('Quantidade de Transações',fontsize=12)
plt.ylabel('Órgãos',fontsize=12)
plt.show()

#%%
