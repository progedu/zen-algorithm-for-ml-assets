from typing import List

import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    特徴量を追加するパイプラインのステップ
    Args:
        column_names (List): 最終的なカラム名のリスト
    """

    def __init__(self, column_names: List):
        self.column_names = column_names

    def fit(self, X: ndarray, y=None):
        # 何も学習しないため処理は記述しない
        return self

    def transform(self, X: ndarray) -> pd.DataFrame:
        """新たな特徴量を追加

        Args:
            X (ndarray): 新たに追加する特徴量
        Returns:
            pd.DataFrame: 特徴量追加後のデータ
        """
        X = pd.DataFrame(X, columns=self.column_names)

        # 一世帯当たりの部屋数, ベッドルーム数
        X["rooms_per_household"] = X["total_rooms"] / X["households"]
        X["bedrooms_per_household"] = X["total_bedrooms"] / X["households"]
        # 一世帯当たりの人数
        X["population_per_household"] = X["population"] / X["households"]
        # ベッドルーム一つあたりの部屋数
        X["rooms_per_bedroom"] = X["total_rooms"] / X["total_bedrooms"]
        # 相対的な左下度合い
        X["degree_of_bottomleft"] = X["latitude"] + X["longitude"]
        return X
