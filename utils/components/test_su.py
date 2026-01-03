import torch
import numpy as np

# python test_su.py

def adjust_different(u, s, alpha=1.0, beta=1.0):
    """
    根据 s 修正 u。

    参数:
        u (torch.Tensor): 节点特征，形状为 [N, D]，值在 [-1, 1] 之间。
        s (torch.Tensor): 序列特征，形状为 [N, D]，值在 [-1, 1] 之间。
        alpha (float): 控制修正强度的超参数。
        beta (float): 控制锐化强度的超参数。

    返回:
        torch.Tensor: 修正后的节点特征。
    """
    # 计算差异
    D = s - u  # 形状为 [N, D]

    # 计算修正权重
    w = torch.sigmoid(alpha * torch.abs(D))  # 形状为 [N, D]，值在 [0, 1] 之间

    # 修正 u
    u_adjusted = u + w * D  # 当差异大时，w 接近 1，u 更接近 s

    # 锐化操作
    u_sharpened = torch.tanh(beta * u_adjusted)  # 当 beta 较大时，输出更接近 1 或 -1

    return u_sharpened


def adjust_absolute(X_node, X_seq, alpha=5.0, beta=3.0, gamma=2.0):
    """
    根据 X_seq (s)的大小修正 X_node (u)。

    参数:
        X_node (torch.Tensor): 节点特征，形状为 [N, D]，值在 [-1, 1] 之间。 u
        X_seq (torch.Tensor): 序列特征，形状为 [N, D]，值在 [-1, 1] 之间。 s
        alpha (float): 控制修正强度的超参数。
        beta (float): 控制锐化强度的超参数。

    返回:
        torch.Tensor: 修正后的节点特征。
    """
    # 计算 X_seq 的绝对值
    abs_X_seq = torch.abs(X_seq)  # 形状为 [N, D]

    # 计算权重：当 X_seq 的绝对值较大时，w_sharp 接近1；较小时接近0
    w_sharp = torch.sigmoid(alpha * (abs_X_seq -  1/106))  # 调整阈值 1/106 0.0094 0.001//  0.5 0.2

    # 尖锐化 X_node 的原始值
    X_node_sharpened = torch.tanh(beta * X_node)

    # 向0调整：当 abs_X_seq 较小时，强制混合0值
    w_zero = torch.sigmoid(gamma * (0.007 - abs_X_seq))  # 0.002 0.2 默认按元素相乘，abs_X_seq越小，w_zero越大
    # sig0 = 0.5 danzeng 如果s0.008，
    X_node_zero_adjusted = X_node * (1 - w_zero) + 0.0 * w_zero

    # print("=====size.w_zero", w_zero.size()) # ([4, 2])
    # print("=====size. X_node",  X_node.size()) # ([4, 2])
    # 修正 X_node：结合尖锐化和向0调整
    u = w_sharp * X_node_sharpened + (1 - w_sharp) * X_node_zero_adjusted

    return u


# 线性插值，不可，正后的u _0406: tensor([1.0000e+00, 2.5219e-04, 3.2562e-05, 7.9110e-06, 6.6687e-01, 8.9851e-03])
def adjust_u_0406_1(u, s, threshold=1/106):
    max_s = torch.max(s)
    u_new = torch.zeros_like(u)
    for i in range(len(u)):
        if s[i] > threshold:
            delta = s[i] - threshold
            scale = delta / (max_s - threshold)
            u_new[i] = u[i] + scale * (torch.sign(u[i]) - u[i])
        else:
            delta = threshold - s[i]
            scale = delta / threshold
            u_new[i] = u[i] * (1 - scale)
    return u_new


def adjust_u_0406_2(u, s, threshold=0.0094, factor=0.55):
    """
    向量化版本，支持输入形状 [batch_size, 106]
    u: 输入张量，形状 [batch_size, 106]
    s: 输入张量，形状 [batch_size, 106]
    """
    u_new = u.clone()
    # 确保所有操作为out-of-place
    max_s = torch.max(s, dim=1, keepdim=True)[0]  # 每个样本单独计算最大值 [B,1]

    # 计算绝对值（不修改原u）
    abs_u = torch.abs(u_new)

    # 生成掩码（不修改原s）
    mask_over = s > threshold

    # 计算超阈值部分的比例（保持维度广播）
    denominator = (max_s - threshold + 1e-7)
    scale_s = torch.where(
       mask_over,
       (s - threshold) / denominator,
       torch.zeros_like(s)
    )

    # 渐进式增量（使用纯函数式操作）
    delta = (1 - abs_u) * scale_s * factor
    new_abs = torch.clamp(abs_u + delta, max=1.0)
    adjusted_over = torch.sign(u_new) * new_abs

    # 处理未超阈值部分（使用where避免原地修改）
    decay_ratio = torch.clamp((threshold - s) / threshold, min=0.0)
    adjusted_under = u_new * (1 - decay_ratio)

    # 合并结果（关键修复：使用torch.where生成新张量）
    u_new_1 = torch.where(mask_over, adjusted_over, adjusted_under)

    return u_new_1



def tanh_adjustment(prob, normalized_weight, epsilon=1e-6):
    """通过双曲正切变换调整概率"""
    prob = prob.cpu().detach().numpy()
    normalized_weight = normalized_weight.cpu().detach().numpy()
    p_clipped = np.clip(prob, -1 + epsilon, 1 - epsilon)
    arctanh_p = np.arctanh(p_clipped)
    scaled = arctanh_p * normalized_weight
    return np.tanh(scaled)

# def adjust_u_0406_2(u, s, threshold=1/106, factor=0.55):
#     max_s = torch.max(s)
#     u_new = u.clone()
#
#     for i in range(len(u)):
#         abs_u = torch.abs(u[i])
#         if s[i] > threshold:
#             # 计算s超阈值的比例
#             scale_s = (s[i] - threshold) / (max_s - threshold + 1e-8)  # 避免除以0
#             # 渐进式增量：原值越大，增量越大，但不超过1
#             delta = (1 - abs_u) * scale_s * factor
#             new_abs = min(abs_u + delta, 1.0)  # 确保不超过1
#             u_new[i] = torch.sign(u[i]) * new_abs
#         else:
#             # 向0衰减逻辑保持不变
#             decay_ratio = (threshold - s[i]) / threshold
#             u_new[i] = u[i] * (1 - decay_ratio)
#     return u_new

# 示例
# u = torch.tensor([[0.5, -0.2], [0.1, 0.8], [-0.1, 0.3], [-0.7, 0.2]], dtype=torch.float32)
# s = torch.tensor([[0.6, -0.1], [0.0, 0.9], [0.8, -0.7], [0.1, 0.9]], dtype=torch.float32)

def scale(p: float, mask_flag=-2): # 要目的是将 (0, 1) 之间的浮点数映射到 (-1, 1) 之间
    # map (0, 1) to (-1, 1)
    if p is not None:
        return (p * 2) - 1
    else:
        return mask_flag
'''
u = torch.tensor([0.10089, 0.00029, 3.82e-5, 9.55e-6, 0.83761, 0.028016])
s = torch.tensor([0.009820476350188255, 0.008203968405723572, 0.008041700348258018, 0.007814893499016762, 0.007510976400226355, 0.0030255813151598])
#  0.0094

att_min = s.min()
att_max = s.max()

mapped_tensor = (s - 1/106) / (att_max - att_min)  # add map
s_scale = scale(mapped_tensor, mask_flag=-2)  # scale 是2* -1  目前是把s也映射到-1，1 再融合
# print("s_scale", s_scale)
# print("mapped_tensor", mapped_tensor)

# u_adjusted_dif = adjust_different(u, s, alpha=1.0, beta=2.0)

# u_adjusted = adjust_absolute(u, s, alpha=5.0, beta=3.0, gamma=2.0)
u_adjusted = adjust_absolute(u, s, alpha=5.0, beta=3.0, gamma=2.0)


u_adjusted_maxmin = adjust_absolute(u, mapped_tensor, alpha=5.0, beta=3.0, gamma=2.0)

u_adjusted_scale = adjust_absolute(u, s_scale, alpha=5.0, beta=3.0, gamma=2.0)
#
print("原始的 u:")
print(u)

print("修正后的 u_our data:")
print(u_adjusted)


adjusted_u = adjust_u_0406_1(u, s)
print("修正后的u _0406:", adjusted_u)


adjusted_u_2 = adjust_u_0406_2(u, s)
print("修正后的u _0406_2:", adjusted_u_2)
# 修正后的u _0406_2: tensor([5.9539e-01, 2.5219e-04, 3.2562e-05, 7.9110e-06, 6.6687e-01, 8.9851e-03])

'''

# print("u_adjusted_maxmin")
# print(u_adjusted_maxmin)
#
# print("修正后的 u_old_data_scale:")
# print(u_adjusted_scale)
# 可以的
# diff修正后的 u:
# tensor([[ 0.8023, -0.2867],
#         [ 0.0947,  0.9360],
#         [ 0.7931, -0.6973],
#         [-0.2877,  0.8706]])

# abso_1 修正后的u:  没有用u的原始值，还是不太好
# tensor([[ 0.9372, -0.1846],
#         [ 0.0000,  0.9905],
#         [ 0.9822, -0.9666],
#         [ 0.1846,  0.9905]])

# abso_2 修正后的u:  用了u的原始值  不对！
# tensor([[ 0.9453, -0.3912],
#         [ 0.1489,  0.9909],
#         [ 0.9820, -0.9649],
#         [-0.5414,  0.9906]])

#  abso_3   -0.3720不符合要求！
# tensor([[ 0.8741, -0.3720],
#         [ 0.1707,  0.9773],
#         [-0.2870,  0.6997],
#         [-0.7362,  0.5322]])

#  abso_4 -0.4026不对！
# tensor([[ 0.8752, -0.4026],
#         [ 0.1957,  0.9764],
#         [-0.2869,  0.6997],
#         [-0.8432,  0.5320]])

#  abso_5  -0.402
# tensor([[ 0.8717, -0.4022],
#         [ 0.1957,  0.9737],
#         [-0.2864,  0.6979],
#         [-0.8419,  0.5314]])

#  abso_6  可！！
# tensor([[ 0.7297, -0.1305],
#         [ 0.0470,  0.9590],
#         [-0.2555,  0.5982],
#         [-0.3485,  0.4962]])

# /原始u s
## tensor([[ 0.5000, -0.2000],   # s[0.6, -0.1]
#         [ 0.1000,  0.8000],    # [0.0, 0.9]
#         [-0.1000,  0.3000],    # [0.8, -0.7]
#         [-0.7000,  0.2000]])   # [0.1, 0.9]