import pickle
from typing import Union
from pandas import DataFrame

def save_as_pickle(obj, path):
    if isinstance(obj, DataFrame):
        obj.to_pickle(path)
    else:
        with open(path,'wb') as f:
            pickle.dump(obj, f)