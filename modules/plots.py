from sklearn.base import BaseEstimator
from numpy.typing import NDArray
from typing import Tuple
import numpy as np
import holoviews as hv
hv.extension("plotly")

def plot_decision_boundary(
    model: BaseEstimator,
    X: NDArray,
    y: NDArray,
    title: str = None,
    xlabel: str = "センサー1",
    ylabel: str = "センサー2",
    background_colors: Tuple = ("#F1F8E9", "#FCE4EC")
):
    """分類境界と散布図を可視化する関数
    Args:
        model (BaseEstimator): 学習済みモデル
        X (NDArray): 2次元の特徴量X
        y (NDArray): 目的変数y
        title (str, optional): グラフのタイトル. Defaults to None.
        xlabel (str, optional): X軸の名称. Defaults to "センサー1".
        ylabel (str, optional): y軸の名称. Defaults to "センサー2".
        background_colors (Tuple, optional): 分類境界背景の色. Defaults to ("#F1F8E9", "#FCE4EC").

    Returns:
        hv.: Holoviewsの可視化オブジェクト
    """
    # マージンを定義
    x1_mergin = X[:, 0].max() * 0.1
    x2_mergin = X[:, 1].max() * 0.1

    # # 背景のグリッドを作成
    x1_min, x1_max = X[:, 0].min() - x1_mergin, X[:, 0].max() + x1_mergin
    x2_min, x2_max = X[:, 1].min() - x2_mergin, X[:, 1].max() + x2_mergin
    dummy_x1 = np.linspace(x1_min, x1_max, 200)
    dummy_x2 = np.linspace(x2_min, x2_max, 200)
    xx1, xx2 = np.meshgrid(dummy_x1, dummy_x2)

    # # 予測を行い、背景色を決定
    pred_input = np.c_[xx1.ravel(), xx2.ravel()]
    dummy_pred = model.predict(pred_input)
    Z = dummy_pred.reshape(xx1.shape)[::-1]    

    # 分類境界の背景とデータポイントをプロット
    color_map = list(background_colors)
    img = hv.Image(data=Z, bounds=(x1_min, x2_min, x1_max, x2_max)).opts(cmap=color_map)

    # 散布図をプロット

    points = hv.Points(data=(X[:, 0], X[:, 1], y), vdims=["z"]).opts(
        color="z", cmap=["blue", "tomato","orange"], show_grid=True
    )
    
    # 全てのholoviewsオブジェクトを結合
    return (img * points).opts(
        title=title, xlabel=xlabel, ylabel=ylabel, width=600, height=400
    )