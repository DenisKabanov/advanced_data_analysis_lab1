import pandas as pd
import numpy as np

def add_early_wakeup(df: pd.DataFrame) -> pd.DataFrame:
    df['Жаворонок'] = -1
    for row in df.index:
        df['Жаворонок'][row] = 1 if int(df['Время пробуждения'][row][0:2]) <= 6 else 0
    df['Жаворонок'] = df['Жаворонок'].astype(np.int8)
    return df

def add_ord_edu(df: pd.DataFrame) -> pd.DataFrame: # вместо one-hot под капотом CatBoost можно сделать порядковый признак (как непрерывная величина, где можно установить threshold), где чем больше - тем лучше (CatBoost умеет работать с категориальными признаками - one-hot, но вдруг приведение к порядковому виду сделает лучше результаты)
    df['Образование_ord'] = df['Образование'].str.slice(0, 1).astype(np.float32)
    return df