import mlrun
import pandas as pd
from sklearn.model_selection import train_test_split
@mlrun.handler(
    outputs=["train_dataset:dataset", "test_dataset:dataset", "label_column"]
)
def data_preparation(dataset: pd.DataFrame, test_size=0.3, label_col="Admit"):
    """A function which preparation the NY taxi dataset

    :param dataset: input dataset dataframe
    :param test_size: the amount (%) of data to use for test

    :return train_dataset, test_dataset, label_column
    """
    print(dataset.columns)
    if test_size != 0:
        train, test = train_test_split(dataset, test_size=test_size)
    else:
        train, test = dataset, dataset
    return train, test, label_col