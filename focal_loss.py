import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.

        Parameters:

        - alpha (list or Tensor): Class weights, length is the number of classes (15). If not provided, the default weight for all classes is 1.

        - gamma (float): Modulation factor, default value is 2.0.

        - reduction (str): Loss reduction method, supports 'mean' (default), 'sum', or 'none'.
        """
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction


        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Focal Loss。

        para:
        - inputs (Tensor): output logits，(batch_size, num_classes)
        - targets (Tensor):  (batch_size,

        return:
        - loss (Tensor):  Focal Loss。
        """

        probs = F.softmax(inputs, dim=1)


        class_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)


        ce_loss = -torch.log(class_probs)


        focal_factor = (1 - class_probs) ** self.gamma


        focal_loss = focal_factor * ce_loss


        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha.gather(0, targets)
            focal_loss = alpha_weight * focal_loss


        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

