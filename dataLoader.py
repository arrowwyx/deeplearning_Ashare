import torch
from torch.utils.data import Dataset
import pandas as pd


class StockDataset(Dataset):
    """
    用于训练的dataloader，输入的x和y就是datadownload中make_window和calc_label的输出
    """
    def __init__(self, x_dataframe, y_dataframe):
        # 去除X和y中包含nan的行
        x_dataframe_drop = x_dataframe.dropna()
        y_dataframe_drop = y_dataframe.dropna()
        # 确保输入的x和y的index对齐
        index_intersection = y_dataframe_drop.index.intersection(x_dataframe_drop.index)
        y_dataframe_drop = y_dataframe_drop.reindex(index_intersection)
        y_dataframe_drop.sort_index(inplace=True)
        x_dataframe_drop = x_dataframe_drop.reindex(index_intersection)
        x_dataframe_drop.sort_index(inplace=True)
        # 将DataFrame转换为PyTorch张量
        self.features = torch.tensor(x_dataframe_drop.values, dtype=torch.float32).reshape(-1, 5, 6)
        self.labels = torch.tensor(y_dataframe_drop.values, dtype=torch.float32)

    def __len__(self):
        # 数据集中样本的数量
        return self.features.shape[0]

    def __getitem__(self, idx):
        # 根据索引idx获取单个样本
        return self.features[idx], self.labels[idx]
