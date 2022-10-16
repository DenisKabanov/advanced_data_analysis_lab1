# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import os
from src.utils import save_as_pickle
from features import *
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir="data/interim/", output_dir="data/preprocessed/"):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    for filename in os.listdir(input_dir):
        if filename.find("data.pkl") != -1 and filename.find("dvc") == -1:
            df = pd.read_pickle(input_dir + filename)
            df = add_ord_edu(df)
            df = add_early_wakeup(df)
            save_as_pickle(df, output_dir + filename)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
