# -*- coding: utf-8 -*-
import click
import logging
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import *
from preprocess import preprocess_data, extract_target
from src.utils import save_as_pickle
import src.config as cfg


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_data_dir', type=click.Path())
@click.argument('output_target_dir', type=click.Path())
def main(input_dir = "data/raw/", output_data_dir="data/interim/", output_target_dir="data/processed/"):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # sys.path.append(os.getcwd()) # на случай, если не будет видеть root проекта
    # 

    for filename in os.listdir(input_dir):
        if filename.find("csv") != -1 and filename.find("dvc") == -1:
            df = pd.read_csv(input_dir + filename)
            df = preprocess_data(df)
            if cfg.TARGET_COLS[0] in df.columns: # проверка, что хотя бы один таргет есть в столбцах рассматриваемого датасета
                df, target = extract_target(df)
                save_as_pickle(target, output_target_dir + filename[0:-4] + "_target.pkl")
            save_as_pickle(df, output_data_dir + filename[0:-4] + "_data.pkl")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
