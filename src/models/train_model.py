# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.utils import save_as_pickle
import pandas as pd
import numpy as np
import src.config as cfg
from src.utils import save_as_pickle

from sklearn.svm import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.compose import *
from sklearn.pipeline import *
from sklearn.metrics import *
from sklearn.impute import *
from sklearn.multioutput import *
from catboost import *


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path())
def main(input_data_filepath="data/processed/train_data.pkl", input_target_filepath="data/processed/train_target.pkl", output_model_filepath="models/model.sav"):

    train_data = pd.read_pickle(input_data_filepath)
    train_target = pd.read_pickle(input_target_filepath)

    real_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler',  StandardScaler())
    ]) 

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='NA')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int8))
    ])

    real = cfg.REAL_COLS
    real.append(cfg.NEW_COLS[1])

    one = cfg.ONE_COLS
    one.append(cfg.NEW_COLS[0])

    preprocess_pipe = ColumnTransformer(transformers=[
        ('real_cols', real_pipe, real),
        ('cat_cols', cat_pipe, cfg.CAT_COLS),
        ('one_cols', 'passthrough', one)
    ])

    Transformer = preprocess_pipe.fit(train_data)

    X_train, X_test, Y_train, Y_test = train_test_split(
        Transformer.transform(train_data),
        train_target,
        train_size=0.8,
        random_state=7,
    )

    train_pool = Pool(X_train, Y_train)
    test_pool  = Pool(X_test, Y_test)

    model = CatBoostClassifier(
        iterations=500, 
        loss_function='MultiLogloss', 
        learning_rate=0.05,
        bootstrap_type='Bayesian',
        leaf_estimation_iterations=5, 
        leaf_estimation_method='Gradient', 
        custom_metric=['Recall', 'F1'],
        random_seed=7,
    )

    model.fit(train_pool, 
            eval_set=test_pool, 
            metric_period=10, 
            plot=True, 
            verbose=True, 
    )

    save_as_pickle(model, output_model_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
