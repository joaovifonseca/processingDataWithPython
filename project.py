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

# pd.options.display.max_rows = 5
dr = pd.read_csv('202201_CPGF.csv', encoding="latin-1", sep=';')
df = pd.DataFrame(dr)
df = df.rename(columns=cabecalhos, inplace=True)
print(df.groupby(by="NOME_ORGAO_SUPERIOR").sum())

# print(df.groupby('ANO_EXTRATO')["VALOR_TRANSACAO"].count())
