import src.config as cfg
import numpy as np
import pandas as pd


def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame: # удаляем лишний столбец
    if 'ID_y' in df.columns:
        df.drop('ID_y', axis=1, inplace=True)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame: # в качестве индексов выставляем id
    if idx_col in df.columns:
        df.set_index(idx_col, inplace=True)
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame: # пропущенные значения 'Пол' заполняем самым частым значением
    most_freq = df['Пол'].value_counts().index[0]
    df['Пол'].fillna(most_freq, inplace=True)
    return df


def fill_other_missed_values(df: pd.DataFrame) -> pd.DataFrame: # помимо одного пропущенного значения 'Пол', в данных имеется 534 пропуска в 'Возраст курения', 546 в 'Сигарет в день', 732 в 'Частота пасс кур', 167 в 'Возраст алког'
    for col in cfg.MANY_MISSING_DATA_COLS: # заполним пропуски медианой
        med = df[col].median()
        df[col].fillna(med, inplace=True)
    return df
        

def cast_types(df: pd.DataFrame) -> pd.DataFrame: # каст типов
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')
    df[cfg.ONE_COLS] = df[cfg.ONE_COLS].astype(np.int8)
    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    if cfg.TARGET_COLS[0] in df.columns: # если хотя бы один столбец таргетов есть в переданном DataFrame, то и их тоже кастим
        df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_unnecesary_id(df)
    df = set_idx(df, cfg.ID_COL)
    df = fill_sex(df)
    # df = fill_other_missed_values(df)
    df = cast_types(df)
    # в виде pipeline
    return df


def extract_target(df: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
    return df, target