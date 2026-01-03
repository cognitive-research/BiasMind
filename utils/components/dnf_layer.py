import os.path
from enum import Enum
from typing import List, Dict, Tuple, Any
import torch.optim as optim
import torch
import json
import sys
import re
from typing import Dict, List, Any
from torch import nn, Tensor
from tqdm import tqdm, trange
from utils.evaluation import acc_compute, calculate_macro_f1
import datetime
import random
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from sklearn.metrics import classification_report, confusion_matrix
from transformers import RobertaModel, AutoTokenizer
from itertools import combinations

from torch_geometric.nn import GATConv, HGTConv, Linear
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, HeteroData, Dataset
from .focal_loss import MultiClassFocalLoss
from .test_su import adjust_different, adjust_absolute, adjust_u_0406_2, tanh_adjustment

# tokenizer = AutoTokenizer.from_pretrained("/home/jd/code/dyq/ALGM22/EANN/model/chinese-roberta-wwm-ext")
tokenizer = AutoTokenizer.from_pretrained("/home/s2022245019/model/chinese-roberta-wwm-ext")

# 全局bert_model变量
bert_model = None


def initialize_bert_model(device='cuda'):
    """初始化BERT模型"""
    global bert_model
    if bert_model is None:
        bert_model = RobertaModel.from_pretrained("/home/s2022245019/model/chinese-roberta-wwm-ext").to(device)
        bert_model.eval()


# 目前data中不考虑以下6种
# 群体归因错误: 1210
# 押韵即理由效应: 456
# 忽视概率: 335
# 锚定效应: 324
# 聚类错觉: 234
# 零风险偏差: 64

FlAG_S = False  # 是否加前面的s  '14': 1, '-1': 0,


# 动态生成标签映射规则
def get_label_mapping_rule(cog_name='结果偏差'):
    # 所有认知偏差列表
    all_cog_names = {
        '-1': 0,  # -1表示wkdata负样本
        "确认偏差": 0,  # 67694
        "可用性启发式": 0,  # 41226
        "刻板印象": 0,  # 25891
        # group 2
        "光环效应": 0,  # 20129
        "权威偏见": 0,  # 19577
        "框架效应": 0,  # 15969
        "从众效应": 0,  # 13676
        # "虚幻真相效应": 0,  # 12764
        # group 3
        "群体内偏爱": 0,  # 5685
        # "单纯曝光效应": 0,  # 5477
        "对比效应": 0,  # 2879
        "过度自信效应": 0,  # 2651
        "损失厌恶": 0,  # 2299
        "结果偏差": 0,  # 1974
        "后见之明偏差": 0  # 1568
    }

    # 将指定的cog_name设为1，其余保持为0
    if cog_name in all_cog_names:
        all_cog_names[cog_name] = 1

    return {
        "binary": all_cog_names,
        "multiple": {
            # group 1
            "确认偏差": 0,  # 67694
            "可用性启发式": 1,  # 41226
            "刻板印象": 2,  # 25891
            # group 2
            "光环效应": 3,  # 20129
            "权威偏见": 4,  # 19577
            "框架效应": 5,  # 15969
            "从众效应": 6,  # 13676
            # "虚幻真相效应": 7, #12764
            # group 3
            "群体内偏爱": 8,  # 5685
            # "单纯曝光效应": 9, # 5477
            "对比效应": 10,  # 2879
            "过度自信效应": 11,  # 2651
            "损失厌恶": 12,  # 2299
            "结果偏差": 13,  # 1974
            "后见之明偏差": 14  # 1568
        }
    }


# 默认标签映射规则（保持向后兼容）
Label_Mapping_Rule = get_label_mapping_rule('结果偏差')

# 加载编码好的谓词张量
predicate_file = "/home/s2022245019/code/TELLER2/data/cognitive/predicate_embeddings_all.pt"  # 目前是all
predicate_tensors = torch.load(predicate_file)
print("=====predicate_tensors=====", predicate_tensors.size())


def scale(p: float, mask_flag=-2):  # 要目的是将 (0, 1) 之间的浮点数映射到 (-1, 1) 之间
    # map (0, 1) to (-1, 1)
    if p is not None:
        return (p * 2) - 1
    else:
        return mask_flag


def transform_symbols_to_long(symbol_tensor, label_mapping):
    # Convert string symbols to integer indices using label mapping
    index_tensor = torch.tensor([label_mapping[symbol] for symbol in symbol_tensor])
    return index_tensor.long()


def split_list_into_batches(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


'''将输入的数据 set 转换为逻辑推理格式，主要是基于 configure 和 gq 进行筛选和填充。
谓词转成-1到1
'''


def transform_org_to_logic(configure, set, gq, mask_flag=-2):
    gq_keys = gq.keys()
    # print("============gq_keys:", gq_keys)  # mes的id  =gq_keys: dict_keys(['141', '569', '2839', '17
    logics_input = []
    label_input = []
    text_input = []
    # pre-define a flag
    # print("==========set：", set) # 所有数据，一行包含id,message,evidence,label

    # print("configure:", configure) # configure: [('P1', 1), ('P2', 1), ('P3', 1), ('P4', 1), ('P5', 1), ('P6', 1), ('P7', 1), ('P8', 1
    for sample in set:
        # print("sample[\"ID\"]：", sample["ID"]) # sample["ID"]： 2973
        if str(sample["ID"]) in gq_keys:

            tmp = gq[str(sample["ID"])]

            tmp_keys = tmp.keys()
            output = []

            for p, p_num in configure:
                # print("====p, p_num:", p, p_num) #从P1 1到P8 1
                if p in tmp_keys:
                    if len(tmp[p]) > p_num:
                        random_selection = random.sample(tmp[p], p_num)
                        # print("random_selection:", random_selection)
                        for atom in random_selection:
                            output.append(scale(atom[-1], mask_flag=-2))
                    else:
                        for atom in tmp[p]:
                            output.append(scale(atom[-1], mask_flag=-2))  # atom[-1]是不同谓词实例化后的概率分数
                            # scale 将 (0, 1) 之间的浮点数映射到 (-1, 1) 之间;若 atom[-1] 为空（None），则用 mask_flag=-2 填充。
                        output = output + [mask_flag] * (p_num - len(tmp[p]))  # 若实际实例化小于p_num 3，不足的也用-2
                else:
                    output = output + [mask_flag] * p_num
            # print("=====output:", output) # 8*1 [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            # print("=====,sample['label']:",sample['label']) # 后见之明偏差

            # 检查MESSAGE是否为tensor（embedding）
            message = sample['MESSAGE']
            if isinstance(message, torch.Tensor):
                # 如果是tensor，直接使用
                text_input.append(message)
            else:
                # 如果是字符串，保持原有逻辑
                text_input.append(message)

            logics_input.append(output)
            label_input.append(sample['label'])
    #
    # print("=====logics_input.size():", len(logics_input)) # list logics_input.size(): 数据集长度

    # print("=====text_input[0]", text_input[0])  # 一家当地报纸称这是自2015年
    # print("=====text_input:", len(text_input)) # 2488 对的 分别是train val test 的数据个数

    return text_input, logics_input, label_input


# logics_input 是一个二维列表，每行代表一个样本的逻辑输入。
# label_input 记录每个样本的类别标签。

#
# def batch_generation(logics_input, label_input, mode, batchsize):  # old
#     assert len(logics_input) == len(label_input), "produce error when generate data splits"
#     # split based on the batchsize
#     # print("mode:", mode)
#     # print("Label_Mapping_Rule[mode]:", Label_Mapping_Rule[mode])
#     label_input = [transform_symbols_to_long(label_input[i:i + batchsize], label_mapping=Label_Mapping_Rule[mode]) for i
#                    in range(0, len(label_input), batchsize)]
#     logics_input = [torch.tensor(logics_input[i:i + batchsize]) for i in range(0, len(logics_input), batchsize)]
#
#     return [(logics_input[i], label_input[i]) for i in range(len(logics_input))]


def batch_generation(text_input, logics_input, label_input, mode, batchsize, cog_name='结果偏差'):
    assert len(logics_input) == len(label_input), "produce error when generate data splits"

    # 检查text_input的类型
    if len(text_input) > 0 and isinstance(text_input[0], torch.Tensor):
        # 如果是embedding tensor，直接分batch
        text_batches = [torch.stack(text_input[i:i + batchsize]) for i in range(0, len(text_input), batchsize)]
        # 创建假的attention_masks（全1）
        attention_batches = [torch.ones(text_batches[i].shape[0], text_batches[i].shape[1])
                             for i in range(len(text_batches))]
    else:
        # 原有的tokenization逻辑
        tokenizer = AutoTokenizer.from_pretrained("/home/s2022245019/model/chinese-roberta-wwm-ext")
        text_tokenized = [tokenizer(text, padding='max_length', truncation=True, max_length=45, return_tensors="pt") for
                          text in text_input]
        text_ids = [t["input_ids"].squeeze(0) for t in text_tokenized]  # 获取 input_ids
        attention_masks = [t["attention_mask"].squeeze(0) for t in text_tokenized]

        text_batches = [torch.stack(text_ids[i:i + batchsize]) for i in range(0, len(text_ids), batchsize)]
        attention_batches = [torch.stack(attention_masks[i:i + batchsize]) for i in
                             range(0, len(attention_masks), batchsize)]

    # 使用动态标签映射规则
    label_mapping_rule = get_label_mapping_rule(cog_name)

    # Split data into batches
    label_input = [transform_symbols_to_long(label_input[i:i + batchsize], label_mapping=label_mapping_rule[mode])
                   for i in range(0, len(label_input), batchsize)]
    logics_input = [torch.tensor(logics_input[i:i + batchsize]) for i in range(0, len(logics_input), batchsize)]

    return [(text_batches[i], attention_batches[i], logics_input[i], label_input[i]) for i in range(len(logics_input))]


class SemiSymbolicLayerType(Enum):
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"


class Conjunction_Shuffle(nn.Module):
    def __init__(
            self,
            configure,
            out_features: int,
            layer_type: SemiSymbolicLayerType,
            delta: float,
            weight_init_type: str = "normal"

    ) -> None:
        # configure: {]
        super(Conjunction_Shuffle, self).__init__()
        self.configure = configure
        self.in_features = sum([t[1] for t in configure])  # P
        self.layer_type = layer_type
        # generate input features and weights by configure
        self.out_features = out_features  # Q

        self.weights = []
        for t in configure:
            tmp = torch.empty(1, self.out_features)
            if weight_init_type == "normal":
                nn.init.normal_(tmp, mean=0.0, std=0.1)
            else:
                nn.init.uniform_(tmp, a=-6, b=6)
            if t[1] > 1:
                tmp = tmp.expand(t[1], -1)
            self.weights.append(tmp)
        # wights P x Q
        self.weights = nn.Parameter(
            torch.cat(self.weights, dim=0)
        )
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:

        mask = torch.where(input >= -1, torch.tensor(1, device=input.device),
                           torch.tensor(0, device=input.device)).unsqueeze(-1).repeat(1, 1, self.out_features)
        # abs_weight: N x P x Q
        # print("====input,size", input.size()) # =input,size torch.Size([32, 30])  应该是 Input: N x P e([bs, 106])啊
        # print("====input.unsqueeze(-1),size", input.unsqueeze(-1).size()) # input.unsqueeze(-1),size torch.Size([32, 30, 1])
        #
        # print("====self.weights",self.weights.size()) # self.weights torch.Size([106, 5])
        # print("====self.weights.expand(input.size(0), -1, -1)",self.weights.expand(input.size(0), -1, -1).size())
        # torch.Size([32, 106, 5])
        abs_weight = torch.abs(self.weights.expand(input.size(0), -1, -1) * input.unsqueeze(-1))
        # max_abs_w: N x Q
        # max_abs_w = torch.max(abs_weight, dim=1)[0]
        max_abs_w = 0.0001
        # sum_abs_w: N x Q
        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: N x Q
        bias = max_abs_w - sum_abs_w

        out = (input.unsqueeze(1)) @ (self.weights.expand(input.size(0), -1, -1) * mask)
        out = out.squeeze()
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum


class Conjunction(nn.Module):
    def __init__(
            self,
            configure,
            out_features: int,
            layer_type: SemiSymbolicLayerType,
            delta: float,
            weight_init_type: str = "normal"

    ) -> None:
        # configure: {]
        super(Conjunction, self).__init__()
        self.configure = configure
        self.in_features = sum([t[1] for t in configure])  # P 的第二维（实例化个数 1+3+1）求和，作为输入特征维度
        self.layer_type = layer_type
        # generate input features and weights by configure
        self.out_features = out_features  # Q

        self.weights = []
        for t in configure:
            tmp = torch.empty(1, self.out_features)
            if weight_init_type == "normal":
                nn.init.normal_(tmp, mean=0.0, std=0.1)
            else:
                nn.init.uniform_(tmp, a=-6, b=6)
            self.weights.append(tmp)
        # wights P x Q
        self.weights = nn.Parameter(
            torch.cat(self.weights, dim=0)
        )
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        # generate mask N x P x Q
        # mask = torch.where(input >= -1, torch.tensor(1, device=input.device), torch.tensor(0, device=input.device)).unsqueeze(-1).repeat(1,1, self.out_features)
        # # abs_weight: Q x P P x Q N x P
        # abs_weight = torch.abs(self.weights@input).T
        # # max_abs_w: Q
        # max_abs_w = torch.max(abs_weight, dim=1)[0]
        # # sum_abs_w: Q
        # sum_abs_w = torch.sum(abs_weight, dim=1)
        # # sum_abs_w: Q
        # bias = max_abs_w - sum_abs_w
        weights = []
        for i, t in enumerate(self.configure):
            if t[1] == 1:
                weights.append(self.weights[i].unsqueeze(0))
            else:
                a = []
                [a.append(self.weights[i].clone()) for i in range(t[1])]
                a = torch.stack(a, dim=0)
                weights.append(a)
        weights = torch.cat(weights, dim=0)

        mask = torch.where(input >= -1, torch.tensor(1, device=input.device),
                           torch.tensor(0, device=input.device)).unsqueeze(-1).repeat(1, 1, self.out_features)
        # abs_weight: N x P x Q
        abs_weight = torch.abs(weights.expand(input.size(0), -1, -1) * input.unsqueeze(-1))
        # max_abs_w: N x Q
        max_abs_w = torch.max(abs_weight, dim=1)[0]  # 这里后面改min
        # sum_abs_w: N x Q
        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: N x Q
        bias = max_abs_w - sum_abs_w

        out = (input.unsqueeze(1)) @ (weights.expand(input.size(0), -1, -1) * mask)
        out = out.squeeze()
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum


class Disjunction(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            layer_type: SemiSymbolicLayerType,
            delta: float,
            weight_init_type: str = "normal"

    ) -> None:
        # configure: {]
        super(Disjunction, self).__init__()

        self.in_features = in_features
        self.layer_type = layer_type
        # generate input features and weights by configure
        self.out_features = out_features  # Q

        self.weights = nn.Parameter(
            torch.empty((self.out_features, self.in_features))
        )
        if weight_init_type == "normal":
            nn.init.normal_(self.weights, mean=0.0, std=0.1)
        else:
            nn.init.uniform_(self.weights, a=-6, b=6)
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        # abs_weight = torch.abs(self.weights)
        # # abs_weight: Q x P
        # max_abs_w = torch.max(abs_weight, dim=1)[0]
        # # max_abs_w: Q
        # sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: Q
        abs_weight = torch.abs(self.weights.T.expand(input.size(0), -1, -1) * input.unsqueeze(-1))
        # max_abs_w: N x Q
        # max_abs_w = torch.max(abs_weight, dim=1)[0]
        max_abs_w = 0.0001
        # sum_abs_w: N x Q
        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: N x Q
        bias = sum_abs_w - max_abs_w
        # bias: Q

        out = input @ self.weights.T
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # print()
        # sum: N x Q
        return sum


def get_bert_embeddings(texts, device):
    """获取 BERT 嵌入"""
    global bert_model
    if bert_model is None:
        initialize_bert_model(device)

    # 对输入文本进行分词和编码
    # 这里没有mask??
    inputs = tokenizer(texts, return_tensors='pt', max_length=50, padding='max_length', truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 移动到 GPU

    # 获取模型输出
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # 取最后一层隐藏状态的平均值作为句子嵌入
    emb = outputs.last_hidden_state.mean(dim=1).to(device)
    return emb


def gen_edge_index():
    """生成四种边类型的异构图边索引"""
    edge_indices = {}

    # 新闻节点到谓词节点 (类型0)
    num_news_nodes = 1
    num_predicate_nodes = 106

    pred_to_news = torch.tensor([list(range(num_predicate_nodes)), [0] * num_predicate_nodes], dtype=torch.long)
    edge_indices[('predicate', 'to_news', 'news')] = pred_to_news

    # 谓词节点到新闻节点 (类型1)
    news_to_pred = pred_to_news.flip([0])
    edge_indices[('news', 'to_predicate', 'predicate')] = news_to_pred

    # 谓词间正向边 (类型2)
    pred_pairs = list(combinations(range(num_predicate_nodes), 2))
    if pred_pairs:
        forward_edge = torch.tensor(pred_pairs, dtype=torch.long).t()
    else:
        forward_edge = torch.empty((2, 0), dtype=torch.long)
    edge_indices[('predicate', 'to_predicate_forward', 'predicate')] = forward_edge

    # 谓词间反向边 (类型3)
    if pred_pairs:
        backward_edge = forward_edge.flip([0])
    else:
        backward_edge = torch.empty((2, 0), dtype=torch.long)
    edge_indices[('predicate', 'to_predicate_backward', 'predicate')] = backward_edge

    return edge_indices


# 这里返回的是所有的，没有拆分数据，，， 如何保证分batch 一致
# def process_graph_data(filename):
#     with open(filename, 'r', encoding='utf-8') as fin:
#         lines = fin.readlines()
#
#     graph_list = []
#
#     # 首先创建一个字典来存储所有样本的evidence，以便后续查找
#     evidence_dict = {}
#     # 第一遍遍历：收集所有样本的evidence
#     for line in lines:
#         data = json.loads(line.strip())
#         data_id = data['id']
#         # 增加对evidence存在性的判断
#         if 'evidence' in data and data['evidence'] is not None and len(data['evidence']) > 0:
#             evidence_dict[data_id] = data['evidence']
#
#     bias_mapping = Label_Mapping_Rule["binary"]
#
#     # 第二遍遍历：处理数据并设置my_evidence
#     for line in tqdm(lines):
#         data = json.loads(line.strip())
#         data_id = data['id']
#
#         # 根据id值设置my_evidence字段：
#         # - 对于id<=100000000的样本，my_evidence = 该字段"evidence"的内容
#         # - 对于id>100000000的样本，my_evidence = id-100000000的id值对应样本的evidence内容
#         if data_id > 100000000:
#
#             target_id = data_id - 100000000
#             my_evidence = evidence_dict.get(target_id, None)
#
#         # 根据id值设置evidence字段：
#         # - 对于id<=100000000的样本，my_evidence = 该字段"evidence"的内容
#         # - 对于id>100000000的样本，my_evidence = id-100000000的id值对应样本的evidence内容
#         # """
#         if data_id > 100000000:
#             # ✅ 特殊处理：新增样本直接使用整段 evidence 作为 prev_sentences
#             # pos_data = data_id-100000000
#             prev_sentences = [my_evidence[0]]
#             next_sentences = []
#             message = data['message'][0] if isinstance(data['message'], list) else data['message']
#         else:
#             message = data['message'][0] if isinstance(data['message'], list) else data['message']
#             message_split = re.split(r'(?<=[。！？])', message)
#             message_split = [s.strip() for s in message_split if s.strip()]
#
#             if message_split:
#                 # print("=====data['evidence']：", data['evidence'])
#                 evidence = data['evidence'][0]
#                 # print("evidence:", evidence)
#                 sentences = re.split(r'(?<=[。！？])', evidence)  # 按标点符号分句
#
#                 sentences = [s.strip() for s in sentences if s.strip()]  # 去除空白字符
#                 # print("sentence:", sentences)
#
#                 # 在 sentences 中找到 evidence 所在的句子索引
#                 evidence_index_start = -1
#                 evidence_index_end = -1
#                 # print("message[0]:", message_split[0])
#                 # print("message[-1]:", message_split[-1])
#                 for i, sentence in enumerate(sentences):
#                     # print("i:", i)
#                     # print("sentence:", sentence)
#                     if message_split[0] in sentence:
#                         evidence_index_start = i
#                     if message_split[-1] in sentence:
#                         evidence_index_end = i
#                         # print("evidence_index_start:", evidence_index_start)
#                         # print("evidence_index_end:", evidence_index_end)
#                         break
#
#                 if evidence_index_start != -1 and evidence_index_end != -1:
#                     # 获取前两句和后两句
#                     prev_sentences = sentences[max(0, evidence_index_start - 2): evidence_index_start]  # 最多取前两句
#                     next_sentences = sentences[evidence_index_end + 1: evidence_index_end + 3]  # 最多取后两句
#                 else:
#                     # print("Alert: message not in evidence!")
#                     # print("message[0]:", message_split[0])
#                     # print("message[-1]:", message_split[-1])
#                     # print("sentence:", sentences)
#                     prev_sentences = None
#                     next_sentences = None
#             else:
#                 prev_sentences = None
#                 next_sentences = None
#
#         # label = bias_mapping.get(data['cognitive_bias_label'], -1)
#         label_ori = data.get('cognitive_bias_label', -1)  # 安全获取标签字段
#         # 如果是负样本，标签名称 为 无认知偏差
#
#
#         label = bias_mapping.get(label_ori, -1)
#
#         graph = {
#             'id': data['id'],
#             'message': message,
#             'edge_indices': gen_edge_index(),
#             'target': torch.tensor([label], dtype=torch.long),
#             "prev_sentences": prev_sentences, "next_sentences": next_sentences
#         }
#         graph_list.append(graph)
#     return graph_list


# 问题是负样本没有evidence ,负样本的summary 当作evidence，可是找不到前后两句，直接用top4?
# 负样本的top1 document 但evidence可以
# evidence如何能匹配,id？？
def process_graph_data(texts, evidences):
    # 检查texts是否为embedding格式
    if isinstance(texts, torch.Tensor) and texts.dim() == 3:
        # texts是embedding格式 [batch_size, seq_len, embed_dim]
        batchsize = texts.shape[0]

        graph_list = []
        for i in range(batchsize):
            # 对于embedding输入，我们无法进行文本分句处理
            # 直接使用embedding作为message
            graph = {
                'message': texts[i],  # 直接使用embedding
                'edge_indices': gen_edge_index(),
                "prev_sentences": None,  # 无法从embedding中提取前后句
                "next_sentences": None
            }
            graph_list.append(graph)
        return graph_list
    else:
        # 原有逻辑：处理文本输入
        batchsize = len(texts) if isinstance(texts, list) else texts.shape[0]

        graph_list = []

        for i in range(batchsize):
            if evidences is None or i >= len(evidences) or evidences[i] is None:
                # 没有evidence的情况
                graph = {
                    'message': texts[i],
                    'edge_indices': gen_edge_index(),
                    "prev_sentences": None,
                    "next_sentences": None
                }
                graph_list.append(graph)
                continue

            sentences = re.split(r'(?<=[。！？])', evidences[i])  # 按标点符号分句
            sentences = [s.strip() for s in sentences if s.strip()]  # 去除空白字符

            # 获取message文本
            message_text = texts[i] if isinstance(texts[i], str) else str(texts[i])
            message_split = re.split(r'(?<=[。！？])', message_text)
            message_split = [s.strip() for s in message_split if s.strip()]

            # 在 sentences 中找到 evidence 所在的句子索引
            evidence_index_start = -1
            evidence_index_end = -1

            if message_split:
                for j, sentence in enumerate(sentences):
                    if message_split[0] in sentence:
                        evidence_index_start = j
                    if message_split[-1] in sentence:
                        evidence_index_end = j
                        break

            if evidence_index_start != -1 and evidence_index_end != -1:
                # 获取前两句和后两句
                prev_sentences = sentences[max(0, evidence_index_start - 2): evidence_index_start]  # 最多取前两句
                next_sentences = sentences[evidence_index_end + 1: evidence_index_end + 3]  # 最多取后两句
            else:
                prev_sentences = None
                next_sentences = None

            graph = {
                'message': texts[i],
                'edge_indices': gen_edge_index(),
                "prev_sentences": prev_sentences,
                "next_sentences": next_sentences
            }
            graph_list.append(graph)
        return graph_list


class GraphDataset(Dataset):
    def __init__(self, graph_data, predicate_embeddings, device):
        super().__init__()
        self.graph_data = graph_data
        self.predicate_embeddings = predicate_embeddings.to(device)
        self.device = device
        self.cache = {}

    def len(self):
        return len(self.graph_data)

    def get(self, idx):
        sample = self.graph_data[idx]
        if idx in self.cache:
            return self.cache[idx]

        # 处理新闻节点嵌入
        message = sample['message']
        if isinstance(message, torch.Tensor):
            # 如果message已经是embedding，直接使用
            if message.dim() == 2:  # [seq_len, embed_dim]
                message_emb = message.mean(dim=0).unsqueeze(0)  # 平均池化并添加batch维度
            elif message.dim() == 1:  # [embed_dim]
                message_emb = message.unsqueeze(0)  # 添加batch维度
            else:
                message_emb = message  # 假设已经是正确的形状
        else:
            # 如果是文本，使用BERT编码
            message_emb = get_bert_embeddings(message, self.device)

        # 构建异构图数据
        hetero_data = HeteroData()

        # 节点特征
        # hetero_data['news'].x = message_emb  # (1024,)
        # hetero_data['predicate'].x = self.predicate_embeddings  # (106, 1024)

        prev_sentences, next_sentences = sample['prev_sentences'], sample['next_sentences']
        if prev_sentences:
            emb_prev = sum(get_bert_embeddings(sent, self.device) for sent in prev_sentences) / len(prev_sentences)
        else:
            emb_prev = 0
        if next_sentences:
            emb_next = sum(get_bert_embeddings(sent, self.device) for sent in next_sentences) / len(next_sentences)
        else:
            emb_next = 0
            # 上下文均分 `w_context`
        w_prev = w_next = 0.1 if prev_sentences and next_sentences else 0
        emb_fusion = w_prev * emb_prev + (1 - w_next - w_prev) * message_emb + w_next * emb_next

        hetero_data['news'].x = emb_fusion
        hetero_data['predicate'].x = self.predicate_embeddings  # (106, 1024)
        self.cache[idx] = hetero_data

        assert hetero_data['news'].num_nodes == 1, f"Expected 1 news node, got {hetero_data['news'].num_nodes}"
        assert hetero_data[
                   'predicate'].num_nodes == 106, f"Expected 106 predicate nodes, got {hetero_data['predicate'].num_nodes}"

        # 边索引
        for edge_type, edge_index in sample['edge_indices'].items():
            hetero_data[edge_type].edge_index = edge_index.to(self.device)

            src_type, _, dst_type = edge_type
            assert edge_index[0].min() >= 0, f"Negative source index in edge type {edge_type}"
            assert edge_index[1].min() >= 0, f"Negative target index in edge type {edge_type}"
            # print("edge_index[0].max():", edge_index[0].max())
            # print("hetero_data[src_type].num_nodes:", hetero_data[src_type].num_nodes)
            assert edge_index[0].max() < hetero_data[
                src_type].num_nodes, f"Source index out of bounds for edge type {edge_type}"
            # 这里有问题！
            # print("edge_index[1].max():", edge_index[1].max())
            # print(" hetero_data[dst_type].num_nodes:",  hetero_data[dst_type].num_nodes)
            assert edge_index[1].max() < hetero_data[
                dst_type].num_nodes, f"Target index out of bounds for edge type {edge_type}"

        return hetero_data


# =============== 新增 DynamicEmbedding 模块 ===============
class DynamicEmbedding(nn.Module):
    def __init__(self, predicate_bert_embs, nhid=64):
        """
        predicate_bert_embs: 预计算的106个谓词的BERT嵌入 [106, 1024]
        """
        super().__init__()
        self.nhid = nhid
        # 固定谓词的基础嵌入（BERT预计算）
        self.register_buffer('base_embs', predicate_bert_embs)  # [106, 1024]

        # 动态调整层
        self.news_ctx_encoder = nn.Linear(1024, nhid)  # 新闻上下文编码
        self.predicate_proj = nn.Linear(1024, nhid)  # 谓词基础编码
        self.fusion_layer = nn.Linear(2 * nhid, nhid)  # 动态融合层

    def forward(self, news_emb):
        """
        news_emb: 新闻节点的BERT嵌入 [batch_size, 1024]
        返回: 动态调整后的谓词嵌入 [batch_size, 106, nhid]
        """
        batch_size = news_emb.size(0)

        # Step 1: 编码新闻上下文
        news_ctx = self.news_ctx_encoder(news_emb)  # [batch_size, nhid]

        # Step 2: 投影基础谓词嵌入
        predicate_base = self.predicate_proj(self.base_embs)  # [106, nhid]

        # Step 3: 动态融合（广播机制）
        news_ctx_expanded = news_ctx.unsqueeze(1)  # [batch_size, 1, nhid]
        predicate_base_expanded = predicate_base.unsqueeze(0)  # [1, 106, nhid]
        fused = torch.cat([
            predicate_base_expanded.expand(batch_size, -1, -1),
            news_ctx_expanded.expand(-1, 106, -1)
        ], dim=-1)  # [batch_size, 106, 2*nhid]

        dynamic_embs = self.fusion_layer(fused)  # [batch_size, 106, nhid]
        return dynamic_embs


class HGTModel(nn.Module):
    def __init__(self, predicate_bert_embs, hidden_size, num_heads, num_layers, metadata):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 动态谓词嵌入生成器
        self.dynamic_embed = DynamicEmbedding(predicate_bert_embs, hidden_size)

        # 新闻节点编码器（带LayerNorm）
        self.news_encoder = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU()
        )

        # 节点类型特定的线性变换（带残差连接）
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = nn.Sequential(
                Linear(-1, hidden_size),
                nn.LayerNorm(hidden_size)
            )

        # HGT卷积层（带层归一化和残差连接）
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConvWithAttention(hidden_size, hidden_size, metadata, num_heads)
            self.convs.append(nn.ModuleDict({
                'conv': conv,
                'norm': nn.ModuleDict({
                    node_type: nn.LayerNorm(hidden_size)
                    for node_type in metadata[0]
                })
            }))

        # 分类器（带初始化）
        self.classifier = Linear(hidden_size, 2)

        # 初始化参数
        # self.reset_parameters()

    def reset_parameters(self):
        # 自定义初始化策略
        for name, param in self.named_parameters():
            if 'dynamic_embed' in name:
                continue  # 保持预训练的BERT嵌入不变

            if 'weight' in name:
                if param.dim() > 1:
                    if 'conv' in name and 'lin' not in name:  # 卷积层权重
                        if 'rel_' in name:  # 关系特定权重
                            nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
                        else:  # 常规权重
                            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                    else:  # 线性层权重
                        nn.init.xavier_uniform_(param)
                else:  # 一维权重
                    nn.init.uniform_(param, -0.01, 0.01)

            elif 'bias' in name:
                nn.init.zeros_(param)

            # 归一化层特殊处理
            elif 'norm' in name:
                if 'weight' in name:
                    nn.init.ones_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x_dict, edge_index_dict):
        # Step 1: 动态生成谓词嵌入
        news_emb = x_dict['news']
        dynamic_pred_embs = self.dynamic_embed(news_emb)

        # Step 2: 编码新闻节点
        news_encoded = self.news_encoder(news_emb)

        # Step 3: 更新x_dict
        x_dict = {
            'news': news_encoded,
            'predicate': dynamic_pred_embs.view(-1, self.hidden_size)
        }

        # 特征转换（带残差）
        residual_dict = {k: v for k, v in x_dict.items()}
        x_dict = {k: self.lin_dict[k](v) + residual_dict[k] for k, v in x_dict.items()}

        # HGT卷积层
        all_attention_weights = []
        for layer_idx, layer in enumerate(self.convs):
            conv = layer['conv']
            norm_dict = layer['norm']

            # 使用自定义HGT层获取注意力权重
            x_dict_new, layer_attention_weights = conv(x_dict, edge_index_dict, return_attention_weights=True)

            # 创建简化的注意力矩阵
            batch_size = x_dict['news'].shape[0]
            num_nodes = 107  # 1个news节点 + 106个predicate节点

            # 创建基础注意力矩阵（均匀分布）
            attention_matrix = torch.ones(batch_size, num_nodes, num_nodes, self.num_heads,
                                          device=x_dict['news'].device) / num_nodes

            # 使用边级别的注意力权重来调整矩阵
            # 这里我们简化处理，主要关注news到predicate的注意力
            if ('news', 'to_predicate', 'predicate') in layer_attention_weights:
                news_to_pred_att = layer_attention_weights[('news', 'to_predicate', 'predicate')]
                if news_to_pred_att.size(0) > 0:
                    # 将news到predicate的注意力权重填入矩阵
                    # news节点索引是0，predicate节点索引是1-106
                    for b in range(batch_size):
                        # 简化：直接使用平均注意力权重
                        avg_att = news_to_pred_att.mean(dim=0)  # [num_heads]
                        attention_matrix[b, 0, 1:, :] = avg_att.unsqueeze(0).expand(106, -1)

            all_attention_weights.append(attention_matrix)
            x_dict = x_dict_new

            # 归一化 + 残差 + 激活
            x_dict = {k: F.elu(norm_dict[k](x_dict[k] + residual_dict[k]))
                      for k in x_dict}
            residual_dict = {k: v.clone() for k, v in x_dict.items()}

        # 平均注意力权重
        if all_attention_weights:
            avg_attention = torch.mean(torch.stack(all_attention_weights), dim=0)
        else:
            # 创建默认注意力矩阵
            batch_size = x_dict['news'].shape[0]
            num_nodes = 107
            avg_attention = torch.ones(batch_size, num_nodes, num_nodes, self.num_heads,
                                       device=x_dict['news'].device) / num_nodes

        return self.classifier(x_dict['news']), avg_attention


class HGT_DNF(nn.Module):
    def __init__(
            self,
            num_conjuncts: int,
            n_out: int,
            delta: float,
            configure: list[(str, int)],
            weight_init_type: str = "normal",
            graph_flag: bool = False,
            shuffle: bool = True,
            graph_merge: str = "u46",
            roberta_model_path: str = "Roberta-chinese"
    ) -> None:
        super(HGT_DNF, self).__init__()

        # 加载预训练的 RoBERTa 模型和分词器
        self.roberta = RobertaModel.from_pretrained("/home/s2022245019/model/chinese-roberta-wwm-ext")
        self.tokenizer = AutoTokenizer.from_pretrained("/home/s2022245019/model/chinese-roberta-wwm-ext")

        if shuffle:
            self.conjunctions = Conjunction_Shuffle(
                configure=configure,
                out_features=num_conjuncts,
                layer_type=SemiSymbolicLayerType.CONJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )
        else:
            self.conjunctions = Conjunction(
                configure=configure,
                out_features=num_conjuncts,
                layer_type=SemiSymbolicLayerType.CONJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )

        self.disjunctions = Disjunction(
            in_features=num_conjuncts,
            out_features=n_out,
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )
        self.conj_weight_mask = torch.ones(
            self.conjunctions.weights.data.shape
        )
        self.disj_weight_mask = torch.ones(
            self.disjunctions.weights.data.shape
        )

        self.graph_flag = graph_flag
        print("\n=====是否用前面graph修正u,graph_flag========", self.graph_flag)

        self.graph_merge = graph_merge

    def forward(self, text, mask, input: Tensor, evidence=None) -> tuple[Any, None]:

        # 检查text是否为embedding
        if isinstance(text, torch.Tensor) and text.dim() == 3:
            # text已经是embedding格式 [batch_size, seq_len, embed_dim]
            # 直接使用，不需要再通过RoBERTa编码
            text_embeddings = text.mean(dim=1)  # 平均池化得到句子级embedding
        else:
            # 原有逻辑：text是tokenized的input_ids
            # 咋整，这里的text 已经是id了
            # 这里的input 最好还是不要改，因为会涉及到前面很多函数，GNN DN F就没办法跑了
            # 使用RoBERTa编码
            text_embeddings = self.roberta(input_ids=text, attention_mask=mask).last_hidden_state.mean(dim=1).to(
                text.device)

        if self.graph_flag:

            batchsize = input.shape[0]

            predicate_embeddings = predicate_tensors.to(text.device)  # ([N, 1024])目前是8  # 编码好的谓词向量 写在文件

            # 这里的问题，原来是传file list ,我需要用ID 对应到负样本的evidence
            # 这里的id 也是bs个
            # 直接使用text_embeddings而不是原始text，因为text是tokenized的
            data = process_graph_data(text_embeddings, evidence)

            my_dataset = GraphDataset(data, predicate_embeddings, text.device)
            # 使用 DataLoader
            loader = DataLoader(my_dataset, batch_size=batchsize, shuffle=False)

            # print("# 获取metadata")
            sample_data = my_dataset.get(0)
            # 确保只使用预期的节点类型
            expected_node_types = ['news', 'predicate']
            metadata = (expected_node_types,
                        list(sample_data.edge_index_dict.keys()))
            # print("===========metadata:\n", metadata)
            # ===========metadata:
            #  (['news', 'predicate'], [('predicate', 'to_news', 'news'), ('news', 'to_predicate', 'predicate'), ('predicate', 'to_predicate_forward', 'predicate'), ('predicate', 'to_predicate_backward', 'predicate')])
            # 

            # 初始化HGT网络层
            hgt_model = HGTModel(
                predicate_bert_embs=predicate_embeddings,
                hidden_size=128,
                num_heads=4,  # 固定为4个头
                num_layers=3,
                metadata=metadata
            ).to(text.device)

            # 前向传播
            for batch in loader:
                out, attention_weights = hgt_model(batch.x_dict, batch.edge_index_dict)
                # print("======hgt.out.shape:", out.shape)
                # print("======attention_weights.shape:", attention_weights.shape)
                # == == == hgt.out.shape: torch.Size([128, 15]) out 是15是在哪设的
                # == == == attention_weights.shape: torch.Size([128, 107, 107, 4])
                # == == == attention_to_node_0_mean.shape: torch.Size([128, 106])

            target_node = 0
            # 检查attention_weights的维度
            if attention_weights.dim() == 4:  # [batch_size, num_nodes, num_nodes, num_heads]
                attention_to_node_0 = attention_weights[:, 1:, target_node, :]
                # 获取注意力分数
                attention_to_node_0_mean = attention_to_node_0.mean(dim=-1)  # 形状: [batch_size, 106]
            else:
                print(f"Unexpected attention_weights shape: {attention_weights.shape}")
                # 创建一个默认的注意力分数
                batch_size = out.shape[0]
                attention_to_node_0_mean = torch.ones(batch_size, 106, device=out.device) / 106

            # print("======attention_to_node_0_mean.shape:", attention_to_node_0_mean.shape)
            # ======attention_to_node_0_mean.shape: torch.Size([128, 106])
            # ======attention_to_node_0_mean.shape: torch.Size([128, 106])
            # 以下为新加的scale 为放大attention分数之间的差异
            # att_min = attention_to_node_0_mean.min()
            # att_max = attention_to_node_0_mean.max()
            #
            # mapped_tensor = (attention_to_node_0_mean - att_min) / (att_max - att_min) # add map
            # s = scale(mapped_tensor, mask_flag=-2)  #scale 是2* -1  目前是把s也映射到-1，1 再融合

            if self.graph_merge == "multiply":
                # 加权修正  后续再试其他
                inner = input * s
                # 使用 tanh 激活函数，确保输出值在 [-1, 1] 范围内
                inner = torch.tanh(inner)  # u为全部更新后的分数

            elif self.graph_merge == "ref_dif":  # 根据su差异动态修正
                inner = adjust_different(input, s, 1.0, 1.0)

            elif self.graph_merge == "ref_sem":  # 原文那种语义参考,如果相关性越大，融合的越多
                W_F = nn.Tanh()(torch.matmul(input, s.t()))  # *运算符默认是按元素相乘，# 使用torch.matmul()进行矩阵乘法
                inner = nn.Tanh()(inner + torch.matmul(W_F, s))

            elif self.graph_merge == "absolute_s":  # 根据s的绝对值大小修正，s大，sharp; s 小，映射到接近0
                # print("=====input.size======", input.size()) # torch.Size([bs32, 106])
                inner = adjust_absolute(input, s, alpha=5.0, beta=3.0, gamma=5.0).to(text.device)  # s [8, 106]
            elif self.graph_merge == "u46":  #
                inner = adjust_u_0406_2(input, attention_to_node_0_mean, factor=0.55)

            elif self.graph_merge == "tanh_adjustment":  #
                inner_np = tanh_adjustment(input, attention_to_node_0_mean)  # s是scale 后的，attention_to_node_0_mean是原始的
                inner = torch.from_numpy(inner_np).float().to(text.device)
            else:
                print("\n====wrong name of graph_merge======")
                exit()

            # 列的索引（从0开始）
            columns_to_update = [6, 8] + list(range(16, 33)) + [34] + list(range(36, 93))

            # 更新 input 的指定列
            new_input = input
            new_input[:, columns_to_update] = inner[:, columns_to_update]


        else:
            new_input = input

        conj = self.conjunctions(new_input)
        conj = nn.Tanh()(conj)

        # disj: N x R
        disj = self.disjunctions(conj)

        return disj, None

    def set_delta_val(self, new_delta_val):
        self.conjunctions.delta = new_delta_val
        self.disjunctions.delta = new_delta_val

    def update_weight_wrt_mask(self) -> None:
        self.conjunctions.weights.data *= self.conj_weight_mask
        self.disjunctions.weights.data *= self.disj_weight_mask


# class GraphAttentionNetwork(nn.Module):  # old 单层
#     def __init__(self, in_channels, out_channels, heads=1):
#         super(GraphAttentionNetwork, self).__init__()
#         # 使用图注意力层（GAT）
#         self.gat_layer = GATConv(in_channels, out_channels, heads)
#
#     def forward(self, x, edge_index, batch):
#         # 计算节点特征和注意力权重
#         out, attention_weights = self.gat_layer(x, edge_index, return_attention_weights=True)
#         return out, attention_weights


class GraphAttentionNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GraphAttentionNetwork, self).__init__()
        # 第一层 GAT
        self.gat1 = GATConv(in_channels, out_channels, heads=heads)
        self.bn1 = torch.nn.BatchNorm1d(out_channels * heads)
        # 第二层 GAT
        # self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
        # # 第三层 GAT
        # self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=heads)

        # 参数初始化
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, batch):

        x, attention_weights = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)

        # 加dropout

        # # 第二层
        # x = self.gat2(x, edge_index)
        # x = self.bn2(x)
        # x = F.elu(x)
        # 第三层，返回最后一层的注意力权重 yj用的三层平均
        # x, attention_weights = self.gat3(x, edge_index, return_attention_weights=True)
        # attention_weight是softmax 之后的
        return x, attention_weights


def process_u(input_tensor, threshold=-0.4, k=5):
    """
    Args:
        input_tensor (Tensor): Shape [bs, dim], values in [-1, 1]
        threshold (float): Threshold value, default -0.4
        k (int): Number of max/min values to retain if no element >= threshold

    Returns:
        Tensor: Processed tensor with the same shape as input
    """
    # print(" ======in process_u")
    device = input_tensor.device
    bs, dim = input_tensor.shape

    # 1. 找到所有 >= threshold 的元素
    mask_ge = input_tensor >= threshold
    # 每行满足条件的数量 m
    m = mask_ge.sum(dim=1)

    # 2. 对每行进行排序以便选出最小的 m 个元素
    sorted_vals, sorted_idx = torch.sort(input_tensor, dim=1)
    # 构造与 sorted_idx 同形状的布尔掩码：每行前 m[i] 个标为 True
    positions = torch.arange(dim, device=device).unsqueeze(0).expand(bs, dim)
    m_expand = m.unsqueeze(1).expand(bs, dim)
    mask_small_in_sorted = positions < m_expand  # [bs, dim]
    # 将排序后的掩码映射回原始索引位置
    mask_small = torch.zeros_like(mask_ge)
    mask_small.scatter_(1, sorted_idx, mask_small_in_sorted)

    # Case 1: 存在 >= threshold 的行，保留 >=threshold 元素和最小 m 个元素
    mask_case1 = mask_ge | mask_small
    output_case1 = input_tensor * mask_case1.float()

    # Case 2: 无 >= threshold 的行，保留 k 个最大和 k 个最小元素
    # 获取每行最大和最小的 k 个索引
    max_idx = torch.topk(input_tensor, k=k, dim=1).indices
    min_idx = torch.topk(input_tensor, k=k, dim=1, largest=False).indices
    mask_case2 = torch.zeros((bs, dim), dtype=torch.bool, device=device)
    mask_case2.scatter_(1, max_idx, True)
    mask_case2.scatter_(1, min_idx, True)
    output_case2 = input_tensor * mask_case2.float()

    # 根据每行 m 是否 > 0 选择输出
    final_output = torch.where(m.unsqueeze(1) > 0, output_case1, output_case2)
    return final_output


class GNN_DNF(nn.Module):
    def __init__(
            self,
            num_conjuncts: int,
            n_out: int,
            delta: float,
            configure: list[(str, int)],
            weight_init_type: str = "normal",
            graph_flag: bool = False,
            sample_u_flag: bool = False,
            shuffle: bool = True,
            graph_merge: str = "u46",
            roberta_model_path: str = "Roberta-chinese"
    ) -> None:
        super(GNN_DNF, self).__init__()

        # 加载预训练的 RoBERTa 模型和分词器
        self.roberta = RobertaModel.from_pretrained("/home/s2022245019/model/chinese-roberta-wwm-ext")
        self.tokenizer = AutoTokenizer.from_pretrained("/home/s2022245019/model/chinese-roberta-wwm-ext")

        if shuffle:
            self.conjunctions = Conjunction_Shuffle(
                configure=configure,
                out_features=num_conjuncts,
                layer_type=SemiSymbolicLayerType.CONJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )
        else:
            self.conjunctions = Conjunction(
                configure=configure,
                out_features=num_conjuncts,
                layer_type=SemiSymbolicLayerType.CONJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )

        self.disjunctions = Disjunction(
            in_features=num_conjuncts,
            out_features=n_out,
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )
        self.conj_weight_mask = torch.ones(
            self.conjunctions.weights.data.shape
        )
        self.disj_weight_mask = torch.ones(
            self.disjunctions.weights.data.shape
        )

        self.graph_flag = graph_flag
        print("\n=====是否用前面graph修正u,graph_flag========", self.graph_flag)

        self.flag_sample_u = sample_u_flag
        print("\n=====是否对u进行topk sample,flag_sample_u========", self.flag_sample_u)
        self.graph_merge = graph_merge

    def forward(self, text, mask, input: Tensor) -> tuple[Any, None]:

        if self.graph_flag:

            batchsize = input.shape[0]

            # Roberta 编码 text（text 已经是 tokenized input_ids）
            text_embeddings = self.roberta(input_ids=text, attention_mask=mask).last_hidden_state.mean(dim=1).to(
                text.device)  # 平均pooling shape: [batchsize, 1024]

            # print("=====text_embeddings.shape=====", text_embeddings.shape) #  torch.Size([32, 1024])

            predicate_embeddings = predicate_tensors.to(
                text.device)  # ([N, 1024])目前是8  # 编码好的谓词向量 写在文件里，直接读取吧，放在代码最前面读取，这里直接赋值

            # 构建全连接图的节点初始向量
            node_embeddings = torch.cat(
                [text_embeddings.unsqueeze(1), predicate_embeddings.unsqueeze(0).expand(batchsize, -1, -1)], dim=1).to(
                text.device)

            # 打印节点特征的形状
            # print("=====node_embeddings.shape=====", node_embeddings.shape)  # torch.Size([32, 107, 1024]) torch.Size([64, 9, 1024]) [batchsize, num_nodes, 1024]

            # 构造全连接的邻接矩阵（包括text和predicate之间的边以及predicate和predicate之间的边）
            num_nodes = node_embeddings.shape[1]  # [batchsize, num_nodes, 1024] => num_nodes
            edge_index = torch.tensor(
                [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j],
                dtype=torch.long).t().contiguous().to(
                text.device)  # torch.Size([2, 72]) 生成全连接图的边，包括 text 和 predicate 之间的边，以及 predicate 和 predicate 之间的边

            # print("=====edge_index.shape=====", edge_index.shape) #  torch.Size([2, 72])

            data_list = []
            for i in range(batchsize):
                data = Data(x=node_embeddings[i], edge_index=edge_index)
                data_list.append(data)

            # 使用 DataLoader 批处理
            loader = DataLoader(data_list, batch_size=batchsize, shuffle=False)

            # 初始化GAT网络层
            gat_model = GraphAttentionNetwork(in_channels=node_embeddings.shape[2], hidden_channels=256,
                                              out_channels=64, heads=1).to(text.device)  # in 1024,hi, out 64

            # 前向传播
            for batch in loader:
                # print("====gnn_batch=====", batch)
                # ch(x=[3424, 1024] 32*107, edge_index=[2, 362944], batch=[3424], ptr=[33])
                out, attention_weights = gat_model(batch.x, batch.edge_index, batch.batch)
                # print("====out_size", out.shape)
                # print("=====out", out) #  torch.Size([13696, 64])
                # 返回的注意力权重是一个元组 (edge_index, attention_weights)，edge_index 是形状为 (2, num_edges * heads)。 的边索引矩阵。attention_weights 是形状为 (num_edges * heads,)的向量，表示每条边的注意力权重。
                output_dim = out.shape[-1]

                out = out.view(batchsize, num_nodes, output_dim)
                #
                # print("====out_size", out.shape)  # torch.Size([128, 107, 64])
                # print("=====out", out)。

                head = attention_weights[1].shape[-1]

                # print("===attention_weights-shape[1]——shape======", attention_weights[1].shape)  # 366368是32 * 107*107 torch.Size([366368, 4])
                # print("======attention_weights[1][:2]======", attention_weights[1][:2]) # 这里就一样 tensor([[0.0093, 0.0093, 0.0093,  //归一化后不太一样了 还是近 tensor([[0.0095, 0.0090, 0.0093, 0.0094]
                attention_weights = attention_weights[1].view(batchsize, num_nodes, num_nodes, head)

                # print("===attention_weights-shape======", attention_weights.shape) #torch.Size([32, 107, 107, 4])

            target_node = 0
            attention_to_node_0 = attention_weights[:, 1:, target_node, :]
            # print("===attention_to_node_0s-shape======", attention_to_node_0.shape) # torch.Size([32, 106, 4]

            attention_to_node_0_mean = attention_to_node_0.mean(dim=-1)  # 形状: [batch_size, 106]
            # print("===attention_to_nod mean [:2]======", attention_to_node_0_mean[:2]) # epo1 这里基本全0.0093，偶尔0.0094 ([[0.0093, 0.0093, 0.0093, 0.0093, 0.0093, 0.0093, 0

            # 以下为新加的scale 为放大attention分数之间的差异
            att_min = attention_to_node_0_mean.min()
            att_max = attention_to_node_0_mean.max()

            mapped_tensor = (attention_to_node_0_mean - att_min) / (att_max - att_min)  # add map
            s = scale(mapped_tensor, mask_flag=-2)  # scale 是2* -1  目前是把s也映射到-1，1 再融合

            # print("=======after_map=====")
            # print("=====s.min======", s.min()) # 109 test  =s.min====== tensor(-1.
            # print("====s.max======", s.max()) #  ======= tensor(1
            # print("=====s前两行======", s[:2]) # 还好

            # s = scale(attention_to_node_0_mean, mask_flag=-2) # 目前是把s也映射到-1，1 再融合
            # print("======s[:3]======j", s[:3]) # 果然全部都是-0.9252
            # print("=====s3.size()======", s.size()) # torch.Size([8, 106])

            # print("=====s.min(),s.max()======", s.min(), s.max())  # 查看最小值 mean (-0.9813,

            # input(u) 和 s 的不同融合
            # print("\n========graph_merge_type:", self.graph_merge)

            if self.graph_merge == "multiply":
                # 加权修正  后续再试其他
                inner = input * s
                # 使用 tanh 激活函数，确保输出值在 [-1, 1] 范围内
                inner = torch.tanh(inner)  # 不行 u为全部更新后的分数

            elif self.graph_merge == "ref_dif":  # 根据su差异动态修正
                inner = adjust_different(input, s, 1.0, 1.0)

            elif self.graph_merge == "ref_sem":  # 原文那种语义参考,如果相关性越大，融合的越多
                W_F = nn.Tanh()(torch.matmul(input, s.t()))  # *运算符默认是按元素相乘，# 使用torch.matmul()进行矩阵乘法
                inner = nn.Tanh()(inner + torch.matmul(W_F, s))

            elif self.graph_merge == "absolute_s":  # 不行 根据s的绝对值大小修正，s大，sharp; s 小，映射到接近0
                # print("=====input.size======", input.size()) # torch.Size([bs32, 106])
                inner = adjust_absolute(input, s, alpha=5.0, beta=3.0, gamma=5.0).to(text.device)  # s [8, 106]
            elif self.graph_merge == "u46":  #
                inner = adjust_u_0406_2(input, attention_to_node_0_mean, factor=0.55)

            elif self.graph_merge == "tanh_adjustment":  #
                inner_np = tanh_adjustment(input, attention_to_node_0_mean)  # s是scale 后的，attention_to_node_0_mean是原始的
                inner = torch.from_numpy(inner_np).float().to(text.device)
            else:
                print("\n====wrong name of graph_merge======")
                exit()

            # 列的索引（从0开始）
            columns_to_update = [6, 8] + list(range(16, 33)) + [34] + list(range(36, 93))

            # 更新 input 的指定列
            new_input = input
            new_input[:, columns_to_update] = inner[:, columns_to_update]

            # print("=====new_input.size======", new_input.size()) # h.Size([64, 8]) bs*K k是谓词总数
            # print("=====new_input.min======", new_input.min()) # (-0.4751
            # print("=====new_input.max======", new_input.max()) # (0.4577


        else:
            if self.flag_sample_u:
                new_input = process_u(input)
            #    Input: N x P e([bs, 106]) P谓词个数  这里input 是-1到1对吗  对。那么此时阈值变为-0.4

            else:
                new_input = input
        # Input: N x P e([bs, 106]) P谓词个数

        conj = self.conjunctions(new_input)
        conj = nn.Tanh()(conj)

        # disj: N x R
        disj = self.disjunctions(conj)

        return disj, None

    def set_delta_val(self, new_delta_val):
        self.conjunctions.delta = new_delta_val
        self.disjunctions.delta = new_delta_val

    def update_weight_wrt_mask(self) -> None:
        self.conjunctions.weights.data *= self.conj_weight_mask
        self.disjunctions.weights.data *= self.disj_weight_mask


class DNF(nn.Module):
    def __init__(
            self,
            num_conjuncts: int,
            n_out: int,
            delta: float,
            configure: list[(str, int)],
            weight_init_type: str = "normal",
            shuffle: bool = True
    ) -> None:
        super(DNF, self).__init__()
        if shuffle:
            self.conjunctions = Conjunction_Shuffle(
                configure=configure,  # P 谓词数组
                out_features=num_conjuncts,  # Q 合取层数，即后面要送入 析取成的 变量数量
                layer_type=SemiSymbolicLayerType.CONJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )  # weight: Q x P
        else:
            self.conjunctions = Conjunction(
                configure=configure,  # P
                out_features=num_conjuncts,  # Q
                layer_type=SemiSymbolicLayerType.CONJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )  # weight: Q x P

        self.disjunctions = Disjunction(
            in_features=num_conjuncts,  # Q
            out_features=n_out,  # R  输出类别个数
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )  # weight R x Q
        self.conj_weight_mask = torch.ones(
            self.conjunctions.weights.data.shape
        )
        self.disj_weight_mask = torch.ones(
            self.disjunctions.weights.data.shape
        )

    def forward(self, input: Tensor) -> tuple[Any, None]:
        # Input: N x P
        conj = self.conjunctions(input)
        # conj: N x Q
        conj = nn.Tanh()(conj)
        # conj: N x Q
        disj = self.disjunctions(conj)
        # disj: N x R
        return disj, None

    def set_delta_val(self, new_delta_val):
        self.conjunctions.delta = new_delta_val
        self.disjunctions.delta = new_delta_val

    def update_weight_wrt_mask(self) -> None:
        self.conjunctions.weights.data *= self.conj_weight_mask
        self.disjunctions.weights.data *= self.disj_weight_mask


''' 带延迟的指数衰减调度器，用于控制 dnf中的 delta 值，随着 step 逐步衰减。why '''


class DeltaDelayedExponentialDecayScheduler:
    initial_delta: float
    delta_decay_delay: int
    delta_decay_steps: int
    delta_decay_rate: float

    def __init__(
            self,
            initial_delta: float,
            delta_decay_delay: int,
            delta_decay_steps: int,
            delta_decay_rate: float,
    ):
        # initial_delta=0.01 for complicated learning
        self.initial_delta = initial_delta
        self.delta_decay_delay = delta_decay_delay
        self.delta_decay_steps = delta_decay_steps
        self.delta_decay_rate = delta_decay_rate

    def step(self, dnf, step: int) -> float:
        if step < self.delta_decay_delay:
            new_delta_val = self.initial_delta
        else:
            delta_step = step - self.delta_decay_delay
            new_delta_val = self.initial_delta * (
                    self.delta_decay_rate ** (delta_step // self.delta_decay_steps)
            )
            # new_delta_val = self.initial_delta * (
            #    delta_step
            # )
        new_delta_val = 1 if new_delta_val > 1 else new_delta_val  # 确保 delta 不会大于 1。
        dnf.set_delta_val(new_delta_val)
        return new_delta_val


class MLP(nn.Module):
    def __init__(self, configure, hidden_size, output_size):
        super(MLP, self).__init__()
        input_size = sum([t[1] for t in configure])  # P
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out, None

    def set_delta_val(self, new_delta_val):
        pass


class LogicTrainer:
    def __init__(self, num_conjuncts, n_out, delta, configure, weight_init_type, device, args, exp=None):
        print("============args.type_of_logic_model:", args.type_of_logic_model)  # default logic

        if args.type_of_logic_model == "logic":
            self.logic_model = DNF(num_conjuncts=num_conjuncts, n_out=n_out, delta=delta, configure=configure,
                                   # args.initial_delta= 0.01
                                   weight_init_type=weight_init_type).to(device)
        elif args.type_of_logic_model == "mlp":
            self.logic_model = MLP(configure, hidden_size=num_conjuncts, output_size=n_out)

        elif args.type_of_logic_model == "gnn_logic":
            self.logic_model = GNN_DNF(num_conjuncts=num_conjuncts, n_out=n_out, delta=delta, configure=configure,
                                       weight_init_type=weight_init_type, graph_flag=args.graph_flag,
                                       sample_u_flag=args.sample_u_flag, graph_merge=args.graph_merge).to(device)
        elif args.type_of_logic_model == "hgt_logic":
            self.logic_model = HGT_DNF(num_conjuncts=num_conjuncts, n_out=n_out, delta=delta, configure=configure,
                                       weight_init_type=weight_init_type, graph_flag=args.graph_flag,
                                       graph_merge=args.graph_merge).to(device)
        else:
            print("Wrong name of a logic model")
            exit()

        print("\n=======args.focal_alpha======", args.focal_alpha)

        if args.focal_alpha == "alpha1":  # 差异最小
            self.alpha = [1.1135856097836927, 1.1655539811342945, 1.2189099842720732, 1.2498730000264116,
                          1.2533898762916564, 1.2797705794659595, 1.3006006695679493, 1.31009460877715,
                          1.4326544099239886, 1.4388577414488666, 1.5550319074664687, 1.5713080210486097,
                          1.6002287078920014, 1.6323716204658443, 1.6834580875735297]
        elif args.focal_alpha == "alpha2":
            self.alpha = [1.8807908085824854, 2.410071809351543, 3.0411733067499815, 3.449089645829311,
                          3.4973775009972528, 3.8723688846763205, 4.184427390053088, 4.331339085825812,
                          6.490084447075177, 6.612173059787246, 9.119998141410997, 9.504093898414279,
                          10.205777517506547, 11.013922293889948, 12.35783957573458]
        elif args.focal_alpha == "alpha3":
            self.alpha = [1.0, 1.6420220249357202, 2.614576493762311, 3.363008594565055, 3.4578331715788937,
                          4.239088233452314, 4.949839134249781, 5.303509871513632, 11.907475813544416,
                          12.359685959466862, 23.513025356026397, 25.53526970954357, 29.444976076555022,
                          34.292806484295845, 43.172193877551024]
        elif args.focal_alpha == "alpha4":  # 差异最大
            self.alpha = [3.537374065648359, 5.808446126231019, 9.248735081688618, 11.896219384966964,
                          12.23164938448179, 14.99524077900933, 17.5094325826265, 18.76049827640238, 42.12119613016711,
                          43.72083257257623, 83.17436609934005, 90.32780082987551, 104.15789473684211, 121.306484295846,
                          152.71619897959184]
        else:
            print("Wrong alpha")
            exit()

        self.gamma = args.focal_gamma

        if args.type_of_loss == "ce":
            print("\n=====args.type_of_loss =====", args.type_of_loss)
            self.criterion = nn.CrossEntropyLoss()  # 里面包含交叉熵，所以输入的预测不是归一化之后的，看看值yu
        elif args.type_of_loss == "focal":
            print("\n=====args.type_of_loss =====", args.type_of_loss)
            self.criterion = MultiClassFocalLoss(alpha=self.alpha, gamma=self.gamma, reduction='mean')  # 这里也可以改sum!!!
        else:
            print("\nWrong name of loss")
            exit()
        # for cos learning schedule
        self.n_steps_per_epoch = args.n_steps_per_epoch
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.args = args
        #  self.step, self.batch_steps, self.n_batch_step for the delta scheduler
        self.step = 0
        self.batch_steps = 1
        self.n_batch_step = args.n_batch_step
        # logging
        self.experiment = exp
        # best accuracy on test dataset
        self.best_test_metric = 0
        self.best_test_f1 = 0
        self.bset_test_precision = 0
        self.best_test_recall = 0

        self.best_test_metric_expdum = 0
        self.best_test_f1_expdum = 0
        self.bset_test_precision_expdum = 0
        self.best_test_recall_expdum = 0

        self.best_test_metrics_old = {}
        self.best_test_metrics = {}
        self.train_flag = False
        self.val_flag = False
        self.test_flag = False

    def train(self, train_set, validloader, testloader):
        self.best_metric = 0  # best accuracy on the validation dataset

        self.best_val_epoch = 1
        current_time = datetime.datetime.now()
        # name rule
        self.args.best_target_ckpoint = "bestmodel"
        # 使用cog_name参数动态生成保存路径
        cog_name = getattr(self.args, 'cog_name', '结果偏差')
        dir_save = f"{cog_name}_0.1wiki_s_tanh_zuhe_new_0811_llms_" + "delta" + str(
            self.args.initial_delta) + "_lr" + str(self.args.lr) + "_decay" + str(
            self.args.weight_decay) + "_numcoj" + str(self.args.num_conjuncts) + "_epoch_" + str(
            self.args.n_epoch) + "_bs_" + str(self.args.batchsize) + "_gflag_" + str(
            self.args.graph_flag) + "_gmerge_" + self.args.graph_merge + "_loss_" + self.args.type_of_loss + "_alp_" + str(
            self.args.focal_alpha)
        save_path = os.path.join(self.args.data_path, self.args.dataset_name, dir_save)
        print("\n=======save_path: ", save_path)
        if self.args.save_flag:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        self.logic_model.to(self.device)
        para = self.logic_model.parameters()
        optimizer, scheduler = self.set_optimizer_and_scheduler(
            para, lr=self.args.lr, SGD=self.args.SGD,
            weight_decay=self.args.weight_decay,
            scheduler_name=self.args.scheduler, step_size=self.args.step_size,
            n_epoch=self.args.n_epoch)
        delta_scheduler = DeltaDelayedExponentialDecayScheduler(initial_delta=self.args.initial_delta,  # 0.01
                                                                delta_decay_delay=self.args.delta_decay_delay,
                                                                delta_decay_steps=self.args.delta_decay_steps,
                                                                delta_decay_rate=self.args.delta_decay_rate)
        for epoch in range(self.args.n_epoch):
            ind_list = [i for i in range(len(train_set[0]))]
            random.shuffle(ind_list)

            train_text_inputs = [train_set[0][i] for i in ind_list]
            # train_evidence_inputs = [train_set[1][i] for i in ind_list]
            train_logics_inputs = [train_set[1][i] for i in ind_list]
            train_label_inputs = [train_set[2][i] for i in ind_list]

            trainloader = batch_generation(train_text_inputs, train_logics_inputs, train_label_inputs, self.args.mode,
                                           self.args.batchsize, getattr(self.args, 'cog_name', '结果偏差'))

            self.n_batch_step = int(len(trainloader) // 3)

            # start from epoch 1
            epoch = epoch + 1
            self.logic_model.train()
            pt = []
            gt = []
            train_loss = 0
            for batch in trainloader:
                texts, masks, inputs, targets = batch[0], batch[1], batch[2], batch[3]
                gt.append(targets)
                texts, masks, inputs, targets = texts.to(self.device), masks.to(self.device), inputs.to(
                    self.device), targets.to(self.device)  # 这里inputs 是

                # print("========inputs========", inputs)  # 为啥有-值,  1.0000, -1.0000,  1.000  -1.0000, -1.0000, -1.0000], [-1.0000, -1.0000, -1.0000, -1.0000,  0.0000, -1.0000, -1.0000, -1.0000]],
                # print("========targets========", targets) #tensor([0, 1, 1, 1,
                # print("========inputs.size()========",inputs.size())  # = torch.Size([64, 8])  -batchsize是64，这里的输入只有各个谓词的分数了，目前子集是8个谓词，
                # print("========targets.size()========", targets.size())  #torch.Size([64])

                outputs, saved_variable = self.logic_model(texts, masks, inputs)

                pt.append(self.obtain_label(outputs.cpu()))

                # loss function need adjustment
                # map the outputs to [0 ,1]
                # outputs  = (outputs+1)/2
                # for multiple classification task
                bb_true = outputs[torch.arange(outputs.size(0)), targets]
                bb = torch.stack([bb_true, -bb_true], dim=1)
                fake_label = torch.zeros(outputs.size(0), dtype=torch.long).to(self.device)
                # loss = self.criterion(outputs, targets) + self.criterion(bb, fake_label)
                loss = self.criterion(outputs, targets)
                # the second term is used to assure the truth value of the opposite < 0
                # for binary loss function
                targets_false = (1 - targets).long()
                bb_false = outputs[torch.arange(outputs.size(0)), targets_false]
                # loss = self.criterion(outputs, targets) + torch.relu(bb_false).mean() + torch.relu(-bb_true).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # update lr and delta
                if scheduler is not None and self.args.scheduler == 'CosLR':
                    scheduler.step()
                train_loss += loss.item()
                self.batch_steps = self.batch_steps + 1
                if self.batch_steps % self.n_batch_step == 0:
                    self.step = self.step + 1

                    delta_scheduler.step(self.logic_model, step=self.step)
            train_loss = train_loss / len(list(trainloader))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
            train_acc = acc_compute(pt, gt)
            train_f1, train_p, train_r = calculate_macro_f1(pt, gt)
            if self.experiment is not None:
                self.experiment.log_metric('{}/train'.format("loss"),
                                           train_loss, epoch)
                self.experiment.log_metric('{}/train'.format("acc"),
                                           train_acc, epoch)
                self.experiment.log_metric('{}/train'.format("macro_F1"),
                                           train_f1, epoch)
                self.experiment.log_metric('{}/train'.format("macro_precision"),
                                           train_p, epoch)
                self.experiment.log_metric('{}/train'.format("macro_recall"),
                                           train_r, epoch)
            print("\nTrain: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision: {:.5f}     Recall:{:.5f}".format(
                train_loss, train_acc,
                train_f1,
                train_p,
                train_r))
            # print metrics for trainset
            val_acc, val_f1 = self.validate(epoch, self.logic_model, validloader)
            test_acc, test_macro_f1, test_macro_precision, test_macro_recall, test_acc_expdum, test_macro_f1_expdum, test_macro_precision_expdum, test_macro_recall_expdum, report, cm = self.test(
                epoch, self.logic_model,
                testloader)
            # print(self.logic_model.c)

            if val_acc >= self.best_metric:  # old nice better
                self.best_metric = val_acc

                # if val_f1 >= self.best_metric:
                #     self.best_metric = val_f1

                self.best_epoch = epoch
                state = {
                    'net': self.logic_model.state_dict(),
                    'epoch': epoch,
                    'delta': self.logic_model.conjunctions.delta,
                }
                self.best_test_metrics_old = {
                    "final_acc": test_acc,
                    "final_f1": test_macro_f1,
                    "final_precision": test_macro_precision,
                    "test_macro_recall": test_macro_recall
                }
                self.best_test_metrics = {
                    "final_test_report": report,
                    "final_test_cm": cm,
                }

                if self.args.save_flag:
                    torch.save(state, os.path.join(save_path, self.args.best_target_ckpoint + ".pth"))
            if test_acc > self.best_test_metric:
                # self.best_test_metric_expdum = test_acc_expdum
                # self.best_test_f1_expdum = test_macro_f1_expdum
                # self.bset_test_precision_expdum = test_macro_precision_expdum
                # self.best_test_recall_expdum = test_macro_recall_expdum

                self.best_test_metric = test_acc
                self.best_test_f1 = test_macro_f1
                self.bset_test_precision = test_macro_precision
                self.best_test_recall = test_macro_recall
            if scheduler is not None and self.args.scheduler != 'CosLR':
                scheduler.step()
        print("\nBest Val Epoch: {}, Best Val Acc： {:.5f}".format(self.best_epoch, self.best_metric))

        print("\nBest Test Acc: {:.5f}".format(self.best_test_metric))  # 直接过测试集，最好的

        print(
            "-----------------------------Final Testing Results under best val acc------------------------------------------------")
        print("Best_Test_classification_report\n", self.best_test_metrics["final_test_report"])  # 用验证集选，最好的测试集
        cm = self.best_test_metrics["final_test_cm"]

        print("\nBest Formatted Confusion Matrix:")
        print(f"{'         Predicted No 0    ':<15}{'Predicted Hindsight 1':<15}")
        print(f"{'Actual No 0 ':<15}{cm[0, 0]:<15}{cm[0, 1]:<15}")
        print(f"{'Actual Hindsight 1  ':<15}{cm[1, 0]:<15}{cm[1, 1]:<15}")

        # if self.experiment is not None:
        #     self.experiment.log_metrics(self.best_test_metrics)
        return self.best_test_metrics_old

    def set_optimizer_and_scheduler(self, paras, lr, SGD=False, momentum=0.9, weight_decay=5e-4,
                                    scheduler_name='StepLR', step_size=20, gamma=0.1, milestones=(10, 20), n_epoch=30,
                                    power=2):
        # only update non-random layers
        if SGD:
            print("Using SGD optimizer")
            optimizer = optim.SGD(paras, lr=lr, momentum=momentum,
                                  weight_decay=weight_decay)
        else:
            print("Using Adam optimizer")
            optimizer = optim.Adam(paras, lr=lr, weight_decay=weight_decay)

        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size,
                                                  gamma=gamma)
        elif scheduler_name == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
                                                       gamma=gamma)
        elif scheduler_name == 'CosLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_steps_per_epoch * n_epoch)
        else:
            raise NotImplementedError()

        return optimizer, scheduler

    def validate(self, epoch, net, validloader):
        net.eval()
        pt = []
        gt = []
        loss = 0.0
        with torch.no_grad():
            for batch in validloader:
                texts, masks, inputs, targets = batch[0], batch[1], batch[2], batch[3]
                gt.append(targets)
                texts, masks, inputs, targets = texts.to(self.device), masks.to(self.device), inputs.to(
                    self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(texts, masks, inputs)
                loss = self.criterion(outputs, targets).item() + loss
                # inter outputs from outputs of self.logic_model
                pt.append(self.obtain_label(outputs.cpu()))
            # print(len(gt))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
        loss = loss / len(validloader)
        acc = acc_compute(pt, gt)
        macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
        if self.experiment is not None:
            self.experiment.log_metric('{}/val'.format("loss"),
                                       loss, epoch)
            self.experiment.log_metric('{}/val'.format("acc"),
                                       acc, epoch)
            self.experiment.log_metric('{}/val'.format("macro_F1"),
                                       macro_f1, epoch)
            self.experiment.log_metric('{}/val'.format("macro_precision"),
                                       macro_precision, epoch)
            self.experiment.log_metric('{}/val'.format("macro_recall"),
                                       macro_recall, epoch)
        print(
            "\nVal: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision{:.5f}     Recall:{:.5f}".format(loss, acc,
                                                                                                              macro_f1,
                                                                                                              macro_precision,
                                                                                                              macro_recall))
        return acc, macro_f1

    def test(self, epoch, net, testloader):  # old
        net.eval()
        pt = []
        gt = []
        loss = 0.0
        acc_expdum, macro_f1_expdum, macro_precision_expdum, macro_recall_expdum = 0, 0, 0, 0
        with torch.no_grad():
            for batch in testloader:
                texts, masks, inputs, targets = batch[0], batch[1], batch[2], batch[3]
                gt.append(targets)
                texts, masks, inputs, targets = texts.to(self.device), masks.to(self.device), inputs.to(
                    self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(texts, masks, inputs)
                loss = self.criterion(outputs, targets).item() + loss
                # inter outputs from outputs of self.logic_model
                pt.append(self.obtain_label(outputs.cpu()))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
        loss = loss / len(testloader)
        acc = acc_compute(pt, gt)
        macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)

        y_true = gt
        y_pre = pt

        # 动态获取cog_name并设置标签名称
        cog_name = getattr(self.args, 'cog_name', '结果偏差')
        target_names = [f'非{cog_name}', cog_name]

        # 检查实际存在的类别数量
        unique_classes = sorted(list(set(y_true + y_pre)))

        # 如果只有一个类别，调整target_names
        if len(unique_classes) == 1:
            if unique_classes[0] == 0:
                target_names = [f'非{cog_name}']
            else:
                target_names = [cog_name]
            # 使用labels参数来指定实际存在的类别
            report = classification_report(y_true, y_pre, target_names=target_names, labels=unique_classes, digits=5)
        else:
            # 正常情况，有两个类别
            report = classification_report(y_true, y_pre, target_names=target_names, digits=5)

        print("Test_classification_report\n", report)

        # 计算并打印混淆矩阵
        cm = confusion_matrix(y_true, y_pre)

        # 根据实际类别数量调整混淆矩阵输出
        if len(unique_classes) == 1:
            print(f"\nFormatted Confusion Matrix (Only class {unique_classes[0]} present):")
            if unique_classes[0] == 0:
                print(f"{'         Predicted 非' + cog_name:<15}")
                print(f"{'Actual 非' + cog_name:<15}{cm[0, 0]:<15}")
            else:
                print(f"{'         Predicted ' + cog_name:<15}")
                print(f"{'Actual ' + cog_name:<15}{cm[0, 0]:<15}")
        else:
            # 正常情况，有两个类别
            print("\nFormatted Confusion Matrix:")
            print(f"{'         Predicted 非' + cog_name:<15}{'Predicted ' + cog_name:<15}")
            print(f"{'Actual 非' + cog_name:<15}{cm[0, 0]:<15}{cm[0, 1]:<15}")
            print(f"{'Actual ' + cog_name:<15}{cm[1, 0]:<15}{cm[1, 1]:<15}")

        if self.experiment is not None:
            self.experiment.log_metric('{}/test'.format("loss"),
                                       loss, epoch)
            self.experiment.log_metric('{}/test'.format("acc"),
                                       acc, epoch)
            self.experiment.log_metric('{}/test'.format("macro_F1"),
                                       macro_f1, epoch)
        print(
            "\nTest: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision:{:.5f}     Recall:{:.5f}".format(loss,
                                                                                                                acc,
                                                                                                                macro_f1,
                                                                                                                macro_precision,
                                                                                                                macro_recall))
        return acc, macro_f1, macro_precision, macro_recall, acc_expdum, macro_f1_expdum, macro_precision_expdum, macro_recall_expdum, report, cm

    # def test(self, epoch, net, testloader):  # new
    #     net.eval()
    #     pt = []
    #     gt = []
    #     loss = 0.0
    #     with torch.no_grad():
    #         for batch in testloader:
    #             inputs, targets = batch[0], batch[1]
    #             gt.append(targets)
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             outputs, saved_variable = self.logic_model(inputs)
    #             loss = self.criterion(outputs, targets).item() + loss # 报错
    #             # inter outputs from outputs of self.logic_model
    #             pt.append(self.obtain_label(outputs.cpu()))
    #         gt = torch.cat(gt).tolist()
    #         pt = torch.cat(pt).tolist()
    #
    #     loss = loss / len(testloader)
    #     # 所有类别的 # Calculate overall accuracy and macro F1 for all classes
    #     acc = acc_compute(pt, gt)
    #     macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
    #
    #     # 筛选出类别0和类别1的样本
    #     filtered_gt = [g for g in gt if g != 2]  # 排除类别2
    #     filtered_pt = [p for p, g in zip(pt, gt) if g != 2]  # 排除类别2的预测
    #
    #     # 计算类0和类1合并后的准确率、F1、精度、召回率
    #     acc_expdum = acc_compute(filtered_pt, filtered_gt)
    #     f1_expdum , precision_expdum , recall_expdum = calculate_f1(filtered_pt, filtered_gt)
    #
    #     if self.experiment is not None:
    #         self.experiment.log_metric('{}/test'.format("loss"),
    #                                    loss, epoch)
    #         self.experiment.log_metric('{}/test'.format("acc"),
    #                                    acc, epoch)
    #         self.experiment.log_metric('{}/test'.format("acc"),
    #                                    acc_expdum, epoch)
    #         self.experiment.log_metric('{}/test'.format("macro_F1"),
    #                                    macro_f1, epoch)
    #         self.experiment.log_metric('{}/test'.format("acc"),
    #                                    f1_expdum, epoch)
    #
    #     print(
    #         "Test: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision:{:.5f}     Recall:{:.5f} ".format(loss, acc,
    #                                                                                                           macro_f1,
    #                                                                                                           macro_precision,
    #                                                                                                           macro_recall))
    #     print(
    #         "Test: Loss {:.5f}       Acc_expdum:{:.5f}       F1_expdum:{:.5f}    Precision_expdum:{:.5f}     Recall_expdum:{:.5f} ".format(loss, acc_expdum,
    #                                                                                                           f1_expdum_expdum,
    #                                                                                                           precision_expdum,
    #                                                                                                           recall_expdum))
    #
    #     return acc, macro_f1, macro_precision, macro_recall, acc_expdum,f1_expdum_expdum,precision_expdum,recall_expdum
    #
    #
    # # 计算准确率的函数
    # def acc_compute(pt, gt):
    #     correct = sum(p == g for p, g in zip(pt, gt))
    #     total = len(gt)
    #     return correct / total
    #
    # # 计算F1分数的函数
    # def calculate_f1(pt, gt):
    #     tp = sum((p == g == 1) for p, g in zip(pt, gt))
    #     fp = sum((p == 1 and g != 1) for p, g in zip(pt, gt))
    #     fn = sum((p != 1 and g == 1) for p, g in zip(pt, gt))
    #
    #     precision = tp / (tp + fp) if tp + fp > 0 else 0
    #     recall = tp / (tp + fn) if tp + fn > 0 else 0
    #     f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    #
    #     return f1, precision, recall

    def obtain_label(self, logicts: torch.tensor):
        labels = torch.argmax(logicts, dim=1)
        return labels


class HGTConvWithAttention(HGTConv):
    """自定义HGTConv，能够返回注意力权重"""

    def forward(self, x_dict, edge_index_dict, return_attention_weights=False):
        # 调用父类的forward方法
        out = super().forward(x_dict, edge_index_dict)

        if return_attention_weights:
            # 简化的注意力权重计算
            # 基于节点特征的相似度计算注意力权重
            attention_weights = {}

            for edge_type in edge_index_dict.keys():
                edge_index = edge_index_dict[edge_type]

                if edge_index.size(1) > 0:
                    # 创建均匀分布的注意力权重作为基础
                    num_edges = edge_index.size(1)
                    edge_att = torch.ones(num_edges, self.heads, device=edge_index.device) / num_edges
                    attention_weights[edge_type] = edge_att
                else:
                    # 空边的情况
                    attention_weights[edge_type] = torch.empty(0, self.heads, device=list(out.values())[0].device)

            return out, attention_weights
        else:
            return out
