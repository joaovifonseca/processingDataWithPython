import pandas as pd
cabecalhos = {
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
}

dr = pd.read_csv('202201_CPGF.csv', encoding="latin-1", sep=';')
df = pd.DataFrame(dr)

df['VALOR TRANSAÇÃO'] = df['VALOR TRANSAÇÃO'].str.replace(',', '').astype(float)
df_filtro_val_trans = df.groupby('NOME ÓRGÃO')['VALOR TRANSAÇÃO'].sum()
print(df_filtro_val_trans)
