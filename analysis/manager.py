"""
データ取得やCSV吐き出しの処理をまとめたモジュール
"""
import pandas
import numpy
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+"/"

def get_train():
    """
    訓練データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/train.csv', encoding="UTF-8")

    print("------ TRANING ------")
    print(df.isnull().any().describe())
    return df

def get_test():
    """
    入力データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/test.csv', encoding="UTF-8")

    print("------ TEST ------")
    print(df.isnull().any().describe())
    return df

def to_csv(test, predict):
    """
    CSVで書き出し
    """
    ImageIds = numpy.array(test["ImageId"]).astype(int)
    df = pandas.DataFrame(predict, ImageIds, columns = ["Label"])
    df.to_csv("./result.csv", index_label = ["ImageId"])

