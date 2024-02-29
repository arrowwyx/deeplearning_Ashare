import torch
import torch.nn as nn


class ICLoss(nn.Module):
    """
    实现IC损失函数，即预测值和真实值之间的相关系数取负数
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor1, tensor2):
        mean1 = torch.mean(tensor1)
        mean2 = torch.mean(tensor2)
        diff1 = tensor1 - mean1
        diff2 = tensor2 - mean2
        corr = torch.sum(diff1*diff2)/(torch.sqrt(torch.sum(diff1**2))*torch.sqrt(torch.sum(diff2**2)))

        return -corr
