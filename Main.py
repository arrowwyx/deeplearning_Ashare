import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from Model import RecurrentAndNeural
from dataLoader import StockDataset
from torch.utils.data import DataLoader
from LossFunction import ICLoss


def train_model(dataloader, model, epochs=10,
                learning_rate=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}.")
    model.to(device)
    # 损失函数
    criterion = ICLoss()
    # 优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 训练模式
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(x)
            # 计算损失
            loss = criterion(outputs, y)
            # 后向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            if torch.isnan(loss).item():
                print("Error! Loss equals nan！")
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')

    print('Training complete.')


if __name__ == "__main__":

    # 特征个数
    input_size = 6
    hidden_size = 30
    # 模型输出个数
    num_factors = 1
    # 是否为双向GRU
    bidirectional = False
    batch_size = 64
    GRU = RecurrentAndNeural(input_size, hidden_size, num_factors, bidirectional)

    df_X = pd.read_hdf('Data/2020_2023window.h5')
    df_Y = pd.read_hdf('Data/2020_2023ret_10.h5')
    dataset = StockDataset(df_X, df_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_model(dataloader, model=GRU, epochs=5)
