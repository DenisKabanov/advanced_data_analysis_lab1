# загружаем test dataset с фичами (или test.csv -> make_data_set -> build_features)
# предсказываем с помощью модели из train_model.py на нём значения таргета и сохраняем предсказание в data/predicted

# загружаем train dataset  и val_idx
# оцениваем получившуюся модель (https://github.com/iterative/example-get-started/blob/main/src/evaluate.py)

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle
import pandas as pd
import numpy as np
import src.config as cfg
from src.utils import save_as_pickle
from sklearn.metrics import *

@click.command()
@click.argument('input_model_dir', type=click.Path(exists=True))
@click.argument('input_test_filepath', type=click.Path(exists=True))
def main(input_model_dir="models/", input_test_filepath="data/processed/test_data.pkl"):

    model = pd.read_pickle(input_model_dir + "model.pkl")
    to_predict = pd.read_pickle(input_test_filepath)
    transformer = pd.read_pickle(input_model_dir + "transformer.pkl")
    to_predict = transformer.transform(to_predict)
    result = model.predict(to_predict)

    with open("results/predict.txt", "w") as file:
        for row in result:
            file.write(f"{row}\n")
    
    save_as_pickle(result, "data/predicted/predict.pkl")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
