import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        初始化 Focal Loss。

        参数:
        - alpha (list or Tensor): 类别权重，长度为类别数（15）。如果没有提供，则默认所有类别的权重为 1。
        - gamma (float): 调制因子，默认值为 2.0。
        - reduction (str): 损失缩减方式，支持 'mean'（默认）、'sum' 或 'none'。
        """
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma  # 调制因子，需要手动设置
        self.reduction = reduction

        # 设置 alpha（类别权重）
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)  # 将列表转换为 Tensor
            else:
                self.alpha = alpha  # 直接使用 Tensor
        else:
            self.alpha = None  # 如果没有提供 alpha，则默认所有类别的权重为 1

    def forward(self, inputs, targets):
        """
        计算 Focal Loss。

        参数:
        - inputs (Tensor): 模型的输出 logits，形状为 (batch_size, num_classes)。
        - targets (Tensor): 真实标签，形状为 (batch_size,)，每个值为类别索引（0 到 num_classes-1）。

        返回:
        - loss (Tensor): 计算得到的 Focal Loss。
        """
        # 将 logits 转换为概率分布
        probs = F.softmax(inputs, dim=1)

        # 获取真实类别的概率
        class_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        # 计算交叉熵损失
        ce_loss = -torch.log(class_probs)

        # 计算调制因子 (1 - p_t)^gamma
        focal_factor = (1 - class_probs) ** self.gamma

        # 计算 Focal Loss
        # if torch.isnan(focal_factor).any() or torch.isinf(focal_factor).any():
        #     print("focal_factor contains NaN or Inf values")
        if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
            print("ce_loss contains NaN or Inf values")
        focal_loss = focal_factor * ce_loss

        # 如果提供了 alpha，则乘以类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)  # 确保 alpha 在正确的设备上
            alpha_weight = self.alpha.gather(0, targets)  # 根据真实标签选择对应的 alpha
            focal_loss = alpha_weight * focal_loss

        # 根据 reduction 参数返回损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# alpha_method6 =  [1.0, 1.6420220249357202, 2.614576493762311, 3.363008594565055, 3.4578331715788937, 4.239088233452314, 4.949839134249781, 5.303509871513632, 11.907475813544416, 12.359685959466862, 23.513025356026397, 25.53526970954357, 29.444976076555022, 34.292806484295845, 43.172193877551024]
#
# # 初始化 Focal Loss
# focal_loss = MultiClassFocalLoss(alpha=alpha_method6, gamma=2.0, reduction='mean')
#
# # 模型输出（logits）和真实标签
# # inputs = torch.randn(10, 15)  # batch_size=10, num_classes=15
# # targets = torch.randint(0, 15, (10,))  # 真实标签，每个值为 0 到 14
#
# # 计算损失
# loss = focal_loss(inputs, targets)
# print(loss)