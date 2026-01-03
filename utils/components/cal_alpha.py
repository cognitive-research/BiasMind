import numpy as np


def calculate_alpha_method1(class_counts):
    """
    方法 1：基于样本数量的倒数。
    """
    total_samples = sum(class_counts)
    alpha = [total_samples / count for count in class_counts]
    return alpha

def calculate_alpha_method2(class_counts):
    """
    方法 2：基于样本数量的对数。
    """
    total_samples = sum(class_counts)
    alpha = [np.log(total_samples) / np.log(count) for count in class_counts]
    return alpha

def calculate_alpha_method3(class_counts):
    """
    方法 3：基于类别频率的平方根。
    """
    total_samples = sum(class_counts)
    alpha = [np.sqrt(total_samples / count) for count in class_counts]
    return alpha


def calculate_alpha_method6(class_counts):
    """
    方法 6：基于类别频率的标准化加权。
    """
    max_count = max(class_counts)
    alpha = [max_count / count for count in class_counts]
    return alpha

# 示例数据
class_counts = [67694, 41226, 25891, 20129, 19577, 15969, 13676, 12764, 5685, 5477, 2879, 2651, 2299, 1974, 1568]  # 每个类别的样本数量

# 计算 alpha
alpha_method1 = calculate_alpha_method1(class_counts)
alpha_method2 = calculate_alpha_method2(class_counts)
alpha_method3 = calculate_alpha_method3(class_counts)

alpha_method6 = calculate_alpha_method6(class_counts)

# 打印结果
print("alpha_method1 = ", alpha_method1) # 差异最大4
#alpha_method1 =  [3.537374065648359, 5.808446126231019, 9.248735081688618, 11.896219384966964, 12.23164938448179, 14.99524077900933, 17.5094325826265, 18.76049827640238, 42.12119613016711, 43.72083257257623, 83.17436609934005, 90.32780082987551, 104.15789473684211, 121.306484295846, 152.71619897959184]
print("\nalpha_method2 = ", alpha_method2) # 差异最小1
#alpha_method2 =  [1.1135856097836927, 1.1655539811342945, 1.2189099842720732, 1.2498730000264116, 1.2533898762916564, 1.2797705794659595, 1.3006006695679493, 1.31009460877715, 1.4326544099239886, 1.4388577414488666, 1.5550319074664687, 1.5713080210486097, 1.6002287078920014, 1.6323716204658443, 1.6834580875735297]

print("\nalpha_method3 = ", alpha_method3) # 差异2
# alpha_method3 =  [1.8807908085824854, 2.410071809351543, 3.0411733067499815, 3.449089645829311, 3.4973775009972528, 3.8723688846763205, 4.184427390053088, 4.331339085825812, 6.490084447075177, 6.612173059787246, 9.119998141410997, 9.504093898414279, 10.205777517506547, 11.013922293889948, 12.35783957573458]

print("\nalpha_method6 = ", alpha_method6) # 差异3
# alpha_method6 =  [1.0, 1.6420220249357202, 2.614576493762311, 3.363008594565055, 3.4578331715788937, 4.239088233452314, 4.949839134249781, 5.303509871513632, 11.907475813544416, 12.359685959466862, 23.513025356026397, 25.53526970954357, 29.444976076555022, 34.292806484295845, 43.172193877551024]
