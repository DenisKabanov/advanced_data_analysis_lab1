# загружаем train dataset  и val_idx
# оцениваем получившуюся модель (https://github.com/iterative/example-get-started/blob/main/src/evaluate.py)

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
from sklearn.metrics import *

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
def main(input_dir="models/"):

    model = pd.read_pickle(input_dir + "model.pkl")
    model.best_score_.get('validation')

    test_target_real = pd.read_pickle(input_dir + "divided_test_target.pkl")
    test_data = pd.read_pickle(input_dir + "divided_test_data.pkl")

    test_target_predicted = np.array(model.predict(test_data), dtype=np.int8)

    print(f"Accuracy: {accuracy_score(test_target_predicted, test_target_real)}")
    print(f"Recall: {recall_score(test_target_real, test_target_predicted, average=None, zero_division=0)}")
    print(f"ROC AUC: {roc_auc_score(test_target_real, test_target_predicted, average=None)}")
    print(f"F1 score: {f1_score(test_target_real, test_target_predicted, average=None)}")

    with open("results/metrics.txt", "w") as file:
        file.write(f"Accuracy: {accuracy_score(test_target_predicted, test_target_real)}\n")
        file.write(f"Recall: {recall_score(test_target_real, test_target_predicted, average=None, zero_division=0)}\n")
        file.write(f"AUC: {roc_auc_score(test_target_real, test_target_predicted, average=None)}\n")
        file.write(f"F1 score: {f1_score(test_target_real, test_target_predicted, average=None)}\n")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
