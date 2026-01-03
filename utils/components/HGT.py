import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.nn import Linear

from transformers import BertTokenizer, BertModel, RobertaModel
from itertools import combinations
from tqdm import tqdm
from utils.components.focal_loss import MultiClassFocalLoss
import re
import itertools

#
# from typing import Dict, Optional, Tuple, Union
# import torch
# from torch import Tensor
# from torch_geometric.nn import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.utils import softmax
# from torch_geometric.typing import Metadata, NodeType, EdgeType, SparseTensor
# import torch.nn.functional as F
# import math
#
# import math
# from typing import Dict, List, Optional, Union
#
# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nn import Parameter
# # from torch_sparse import SparseTensor
#
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense import Linear
# from torch_geometric.nn.inits import glorot, ones, reset
# from torch_geometric.typing import EdgeType, Metadata, NodeType
# from torch_geometric.utils import softmax
# from torch_geometric.typing import Metadata, NodeType, EdgeType, SparseTensor
#
#
# def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
#     if len(xs) == 0:
#         return None
#     elif aggr is None:
#         return torch.stack(xs, dim=1)
#     elif len(xs) == 1:
#         return xs[0]
#     else:
#         out = torch.stack(xs, dim=0)
#         out = getattr(torch, aggr)(out, dim=0)
#         out = out[0] if isinstance(out, tuple) else out
#         return out
#
#
#
#
# class HGTConv(MessagePassing):
#     def __init__(
#         self,
#         in_channels: Union[int, Dict[str, int]],
#         out_channels: int,
#         metadata: Metadata,
#         heads: int = 1,
#         group: str = "sum",
#         **kwargs,
#     ):
#         super().__init__(aggr='add', node_dim=0, **kwargs)
#
#         if not isinstance(in_channels, dict):
#             in_channels = {node_type: in_channels for node_type in metadata[0]}
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.group = group
#
#         self.k_lin = torch.nn.ModuleDict()
#         self.q_lin = torch.nn.ModuleDict()
#         self.v_lin = torch.nn.ModuleDict()
#         self.a_lin = torch.nn.ModuleDict()
#         self.skip = torch.nn.ParameterDict()
#         for node_type, in_channels in self.in_channels.items():
#             self.k_lin[node_type] = Linear(in_channels, out_channels)
#             self.q_lin[node_type] = Linear(in_channels, out_channels)
#             self.v_lin[node_type] = Linear(in_channels, out_channels)
#             self.a_lin[node_type] = Linear(out_channels, out_channels)
#             self.skip[node_type] = Parameter(torch.Tensor(1))
#
#         self.a_rel = torch.nn.ParameterDict()
#         self.m_rel = torch.nn.ParameterDict()
#         self.p_rel = torch.nn.ParameterDict()
#         dim = out_channels // heads
#         for edge_type in metadata[1]:
#             edge_type = '__'.join(edge_type)
#             self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
#             self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
#             self.p_rel[edge_type] = Parameter(torch.Tensor(heads))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         reset(self.k_lin)
#         reset(self.q_lin)
#         reset(self.v_lin)
#         reset(self.a_lin)
#         ones(self.skip)
#         ones(self.p_rel)
#         glorot(self.a_rel)
#         glorot(self.m_rel)
#
#     def forward(
#         self,
#         x_dict: Dict[NodeType, Tensor],
#         edge_index_dict: Union[Dict[EdgeType, Tensor],
#                                Dict[EdgeType, SparseTensor]],
#         return_edge_weights: bool = False
#     ) -> Union[Dict[NodeType, Optional[Tensor]],
#                Tuple[Dict[NodeType, Optional[Tensor]], Dict[EdgeType, Tensor]]]:
#         H, D = self.heads, self.out_channels // self.heads
#
#         k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}
#         edge_weights_dict = {}  # 用于存储边的权重
#
#         # Iterate over node-types:
#         for node_type, x in x_dict.items():
#             k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
#             q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
#             v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
#             out_dict[node_type] = []
#
#         # Iterate over edge-types:
#         for edge_type, edge_index in edge_index_dict.items():
#             src_type, _, dst_type = edge_type
#             edge_type_str = '__'.join(edge_type)
#
#             a_rel = self.a_rel[edge_type_str]
#             k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)
#
#             m_rel = self.m_rel[edge_type_str]
#             v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)
#
#             # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
#             out, edge_weights = self.propagate(
#                 edge_index, k=k, q=q_dict[dst_type], v=v,
#                 rel=self.p_rel[edge_type_str], size=None
#             )
#             out_dict[dst_type].append(out)
#             if return_edge_weights:
#                 edge_weights_dict[edge_type] = edge_weights
#
#         # Iterate over node-types:
#         for node_type, outs in out_dict.items():
#             out = group(outs, self.group)
#
#             if out is None:
#                 out_dict[node_type] = None
#                 continue
#
#             out = self.a_lin[node_type](F.gelu(out))
#             if out.size(-1) == x_dict[node_type].size(-1):
#                 alpha = self.skip[node_type].sigmoid()
#                 out = alpha * out + (1 - alpha) * x_dict[node_type]
#             out_dict[node_type] = out
#
#         # if return_edge_weights:
#         #     return out_dict, edge_weights_dict
#         return out_dict, edge_weights_dict
#
#     def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
#                 index: Tensor, ptr: Optional[Tensor],
#                 size_i: Optional[int]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
#         # alpha = (q_i * k_j).sum(dim=-1) * rel
#
#         try:
#             alpha = (q_i * k_j).sum(dim=-1) * rel
#         except RuntimeError as e:
#             print(f"Error occurred in calculating alpha: {e}")
#             print(f"q_i shape: {q_i.shape}")
#             print(f"k_j shape: {k_j.shape}")
#             print(f"rel: {rel}")
#
#         alpha = alpha / math.sqrt(q_i.size(-1))
#         alpha = softmax(alpha, index, ptr, size_i)
#         out = v_j * alpha.view(-1, self.heads, 1)
#
#         # if return_edge_weights:
#         #     return out.view(-1, self.out_channels), alpha
#         return out.view(-1, self.out_channels), alpha
#
#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
#                 f'heads={self.heads})')


import random
import numpy as np
import torch


# 0420 直接融这个

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

predicate = {
    'P1': "情绪语言，表示通过具有强烈情感含义的特定词语和短语来影响受众。",
    'P2': "贴标签，将宣传活动的对象标记为目标受众所恐惧、厌恶，或相反地，钦佩或赞扬的事物。",
    'P3': "挥舞旗帜，利用强烈的民族主义情绪（例如种族、性别、政治偏好）来为行动或理念辩护或宣传，通常强调特定的价值观或愿景。",
    'P4': "简化因果，假设某个问题只有单一原因，而实际上可能有多个原因，包括替罪羊策略，即在未深入调查问题复杂性的情况下，将责任推卸给个人或群体。",
    'P5': "喊口号，使用简短、朗朗上口的短语，可能包含标签化和刻板印象。",
    'P6': "非黑即白，将两种选择作为唯一可能性呈现，而实际上可能存在更多选项。",
    'P7': "诉诸人身，攻击对方个人，而不是回应其论点，利用无关的个人特征来反驳对手或支持自身观点。",
    'P8': "诉诸反复，通过重复强调某观点已经被充分讨论或反对意见已被驳斥，从而避免提供证据。",
    'P9': "议题设置，大众媒体通过报道的方向和数量，对某一特定议题进行强调。",
    'P10': "诉诸权威，仅仅因为某个权威或专家支持某一观点，就宣称该观点为真，而不提供其他证据。",
    'P11': "诉诸恐惧/偏见，通过激发对替代方案的焦虑或恐慌，基于先入为主的判断来寻求支持。",
    'P12': "格言论证，使用特定的词句，使得对该话题的批判性思考和有意义讨论变得困难。",
    'P13': "诉诸转移，引入无关内容，以转移对核心问题的注意力。",
    'P14': "历史引用，使用历史事件或人物来类比当前事件。",
    'P15': "预设立场，使用修辞标记来表达媒体的预设立场。",
    'P16': "诉诸潮流，为了避免社会孤立而倾向于与多数人保持一致。",
    'P17': "美好人生，通过与名人建立联系或描绘吸引人、幸福的人物形象来提升士气。",
    'P18': "戈培尔效应，以微妙和间接的方式传递信息，使其被无意识接受，进而影响心理或行为反应。",
    'P19': "假二难推理，在多种可能性存在的情况下，仅呈现有限的选项。",
    'P20': "单方论证，只呈现支持性的理由，而忽略反对的观点。",
    'P21': "经典条件反射，将两个经常一起出现的刺激联系起来，从而引发条件反射。",
    'P22': "认知失调，持有两个相互矛盾的信念所产生的心理不适，导致个体改变行为或信念以减少紧张感。",
    'P23': "诉诸平民，通过使用目标受众的语言和方式来赢得信任。",
    'P24': "个人崇拜，通过大规模宣传，将某个人塑造成群体敬仰的对象。",
    'P25': "妖魔化，将敌人描绘成完全破坏性和邪恶的形象，通常涉及仇恨、个人攻击和诽谤。",
    'P26': "低落士气，削弱对手的士气，以鼓励投降或叛变。",
    'P27': "独裁，将某种观点呈现为唯一可行的选择，通过消除其他选项来简化决策过程。",
    'P28': "虚假信息，篡改或删除公共记录，以编造关于事件、个人或组织的虚假叙述。",
    'P29': "分而治之，将更大的力量拆分为较小的、力量较弱的部分，以防止统一反对。",
    'P30': "以退为进，先提出一个极端的请求，使其被拒绝，然后再提出一个更合理的请求，以提高成功率。",
    'P31': "粗直语，使用严厉或冒犯性的语言，负面地影响观众的感知。",
    'P32': "委婉语，使用温和或间接的表达方式来替代严厉或不愉快的语言。",
    'P33': "制造幸福感，通过使用愉快或吸引人的事件来提升士气。",
    'P34': "夸大，过度强调或夸大事件或情况。",
    'P35': "虚假指控，提出没有事实依据的指控或主张。",
    'P36': "恐惧、不确定与怀疑（FUD），传播负面、怀疑或虚假的信息来影响感知。",
    'P37': "谎言灌喷，通过多个渠道迅速且反复地传播大量信息，不顾真相或一致性。",
    'P38': "得寸进尺，通过先提出一个小请求，再提出一个更大的请求来进行劝说。",
    'P39': "框架化，通过以不同方式呈现相同的问题，使某个选项看起来更有利。",
    'P40': "煤气灯效应，通过否认、虚假信息和矛盾来操控某人，让其怀疑自己的记忆、感知或理智。",
    'P41': "乱枪打鸟，通过大量的论点压倒对手，其中许多论点是有缺陷或不相关的。",
    'P42': "光辉普照，将积极的概念与信息关联，使其在没有证据的情况下更具吸引力。",
    'P43': "关联谬误，通过无关的关联，论证一种事物的特性适用于另一个事物。",
    'P44': "片面事实，一个部分正确但遗漏关键信息的陈述。",
    'P45': "含糊其辞，使用模糊或不明确的术语，进行无分析或无正当理由的操控。",
    'P46': "接受范围，超出个人或群体接受范围的信息，通常会引发心理抵抗。",
    'P47': "负载性语言，使用具有强烈内涵的词汇来激发情绪反应或利用刻板印象。",
    'P48': "谎言与欺骗，使用虚假或扭曲的信息来为行动或信仰辩护，并促使其被接受。",
    'P49': "社会环境控制，通过同伴或社会压力强制执行对某一理念或事业的遵从，与洗脑和精神控制相关。",
    'P50': "冷处理，在明确否定不可能的情况下，最小化事件或情感的重要性。",
    'P51': "人身攻击，批评或攻击一个人的个人特征，而不是他们的论点。",
    'P52': "形式谬误，一个有缺陷的推理，结论没有逻辑地从前提中得出。",
    'P53': "混淆，故意模糊不清的交流，旨在使观众困惑或排除更广泛的理解。",
    'P54': "操作制约，通过奖励或惩罚影响‘自愿’表现的行为。",
    'P55': "单音谬误，将结果归因于单一原因，而忽视其他贡献因素。",
    'P56': "单一思想，通过简化的论证压制替代观点。",
    'P57': "断章取义，删除陈述的原始上下文，创造误导性的推论。",
    'P58': "暗示性语言，间接传达观点、情感或含义，而不明确声明。",
    'P59': "终结性陈词，一种总结性的语言策略，旨在强化信息并促使观众做出最终行动。",
    'P60': "乐观/悲观倾向或褒贬态度。",
    'P61': "问题引导思考或命令促使行动。",
    'P62': "通过幽默或讽刺制造情感冲突或共鸣。",
    'P63': "确定或不确定的语言。",
    'P64': "统计数据。",
    'P65': "模糊信息来源。",
    'P66': "对比结构。",
    'P67': "强调信息的频繁出现。",
    'P68': "增强品牌、观点或事件的熟悉度。",
    'P69': "推断事件发生后是‘不可避免’的。",
    'P70': "从单一特征推断其他属性。",
    'P71': "强调损失或风险。",
    'P72': "淡化收益或不确定性。",
    'P73': "对比风险和收益的表达。",
    'P74': "描述群体行为。",
    'P75': "区分‘我们与他们’。",
    'P76': "群体压力。",
    'P77': "做出绝对陈述。",
    'P78': "模糊概率。",
    'P79': "行话或行业特定术语。",
    'P80': "隐喻、拟人或视觉隐喻。",
    'P81': "押韵或有节奏的语言。",
    'P82': "被动语态。",
    'P83': "过去完成时来暗示确定性。",
    'P84': "预测未来事件。",
    'P85': "连接词来改变预期。",
    'P86': "将个别案例归纳为群体特征。",
    'P87': "忽视多样性。",
    'P88': "忽视反论点或反例。",
    'P89': "可得性偏差。",
    'P90': "通过将时间与结果联系起来误导。",
    'P91': "关注结果而忽视过程。",
    'P92': "事后添加结果以增强预见感。",
    'P93': "强调紧迫性。",
    'P94': "‘问题-解决结构’的叙事结构。",
    'P95': "‘对比结构’的叙事结构。",
    'P96': "‘英雄之旅结构’的叙事结构。",
    'P97': "‘三幕剧结构’的叙事结构。",
    'P98': "‘框架叙事结构’的叙事结构。",
    'P99': "‘循环叙事结构’的叙事结构。",
    'P100': "‘嵌套结构’的叙事结构。",
    'P101': "‘平行结构’的叙事结构。",
    'P102': "‘损失与收益框架’的叙事框架。",
    'P103': "‘属性框架’的叙事框架。",
    'P104': "‘时间框架’的叙事框架。",
    'P105': "‘行动框架’的叙事框架。",
    'P106': "‘社会比较框架’的叙事框架。"
}


# 假设config包含必要的配置参数
class Config:
    train_data_file = r'/home/sdd/jd/wyj/data/cognitive/train_HGT_6.jsonl'
    test_data_file = '/home/sdd/jd/wyj/data/cognitive/test.jsonl'
    val_data_file = '/home/sdd/jd/wyj/data/cognitive/val.jsonl'
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    save_dir = './HGT_saved_models_ce_dynamic_evidence_seed42_tuning'
    batch_size = 32
    best_model_path = os.path.join(save_dir, 'best_hgt_model.pth')
    nheads = 8
    lr = 1e-4
    weight_decay = 5e-4

    # type_of_loss = "focal"
    type_of_loss = "ce"
    focal_alpha = "alpha3"
    focal_gamma = 2
    epoch = 500
    best_val_acc_model_path = os.path.join(save_dir, 'val_acc_best_model.pth')
    best_val_loss_model_path = os.path.join(save_dir, 'val_loss_best_model.pth')
    best_train_loss_model_path = os.path.join(save_dir, 'train_loss_best_model.pth')


config = Config()

# 创建保存目录
os.makedirs(config.save_dir, exist_ok=True)
# tokenizer = BertTokenizer.from_pretrained('/home/s2024244178/Trust_TELLER-main/bert-base-cased') bert_model =
# BertModel.from_pretrained('/home/s2024244178/Trust_TELLER-main/bert-base-casedbert-base-uncased').to(
# config.device).eval()


from transformers import RobertaModel, AutoTokenizer, BertTokenizer

# 使用 chinese-roberta-wwm-ext 模型
tokenizer = BertTokenizer.from_pretrained('/home/sdd/jd/wyj/Trust_TELLER-main/chinese-roberta-wwm-ext')
bert_model = RobertaModel.from_pretrained('/home/sdd/jd/wyj/Trust_TELLER-main/chinese-roberta-wwm-ext').eval()
bert_model.to(config.device)


def get_bert_embeddings(texts, device):
    """获取 BERT 嵌入"""
    # 对输入文本进行分词和编码
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
    # news_to_pred = torch.tensor([[0] * num_predicate_nodes, list(range(num_predicate_nodes))], dtype=torch.long)
    # edge_indices[('news', 'to_predicate', 'predicate')] = news_to_pred
    #
    # # 谓词节点到新闻节点 (类型1)
    # pred_to_news = news_to_pred.flip([0])
    # edge_indices[('predicate', 'to_news', 'news')] = pred_to_news

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


def process_graph_data(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    graph_list = []
    bias_mapping = {
        "确认偏差": 0, "可用性启发式": 1, "刻板印象": 2,
        "光环效应": 3, "权威偏见": 4, "框架效应": 5, "从众效应": 6, "虚幻真相效应": 7,
        "群体内偏爱": 8, "单纯曝光效应": 9, "对比效应": 10, "过度自信效应": 11, "损失厌恶": 12,
        "结果偏差": 13, "后见之明偏差": 14
    }

    for line in tqdm(lines):
        data = json.loads(line.strip())
        message = data['message'][0] if isinstance(data['message'], list) else data['message']
        message_split = re.split(r'(?<=[。！？])', message)
        message_split = [s.strip() for s in message_split if s.strip()]

        if message_split:

            evidence = data['evidence'][0]
            # print("evidence:", evidence)
            sentences = re.split(r'(?<=[。！？])', evidence)  # 按标点符号分句

            sentences = [s.strip() for s in sentences if s.strip()]  # 去除空白字符
            # print("sentence:", sentences)

            # 在 sentences 中找到 evidence 所在的句子索引
            evidence_index_start = -1
            evidence_index_end = -1
            # print("message[0]:", message_split[0])
            # print("message[-1]:", message_split[-1])
            for i, sentence in enumerate(sentences):
                # print("i:", i)
                # print("sentence:", sentence)
                if message_split[0] in sentence:
                    evidence_index_start = i
                if message_split[-1] in sentence:
                    evidence_index_end = i
                    # print("evidence_index_start:", evidence_index_start)
                    # print("evidence_index_end:", evidence_index_end)
                    break

            if evidence_index_start != -1 and evidence_index_end != -1:
                # 获取前两句和后两句
                prev_sentences = sentences[max(0, evidence_index_start - 2): evidence_index_start]  # 最多取前两句
                next_sentences = sentences[evidence_index_end + 1: evidence_index_end + 3]  # 最多取后两句
            else:
                # print("Alert: message not in evidence!")
                # print("message[0]:", message_split[0])
                # print("message[-1]:", message_split[-1])
                # print("sentence:", sentences)
                prev_sentences = None
                next_sentences = None
        else:
            prev_sentences = None
            next_sentences = None

        label = bias_mapping.get(data['cognitive_bias_label'], -1)

        graph = {
            'id': data['id'],
            'message': message,
            'edge_indices': gen_edge_index(),
            'target': torch.tensor([label], dtype=torch.long),
            "prev_sentences": prev_sentences, "next_sentences": next_sentences
        }
        graph_list.append(graph)
    return graph_list


# =============== 新增 DynamicEmbedding 模块 ===============
class DynamicEmbedding(nn.Module):
    def __init__(self, predicate_bert_embs, nhid=64):
        """
        predicate_bert_embs: 预计算的106个谓词的BERT嵌入 [106, 768]
        """
        super().__init__()
        self.nhid = nhid
        # 固定谓词的基础嵌入（BERT预计算）
        self.register_buffer('base_embs', predicate_bert_embs)  # [106, 768]

        # 动态调整层
        self.news_ctx_encoder = nn.Linear(768, nhid)  # 新闻上下文编码
        self.predicate_proj = nn.Linear(768, nhid)  # 谓词基础编码
        self.fusion_layer = nn.Linear(2 * nhid, nhid)  # 动态融合层

    def forward(self, news_emb):
        """
        news_emb: 新闻节点的BERT嵌入 [batch_size, 768]
        返回: 动态调整后的谓词嵌入 [batch_size, 106, nhid]
        """
        batch_size = news_emb.size(0)
        # print("news_emb.shape:", news_emb.shape)
        # print("batch_size:", batch_size)

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
        message_emb = get_bert_embeddings(sample['message'], self.device)

        # 构建异构图数据
        hetero_data = HeteroData()

        # 节点特征
        # hetero_data['news'].x = message_emb  # (768,)
        # hetero_data['predicate'].x = self.predicate_embeddings  # (106, 768)

        # 标签
        hetero_data.y = sample['target'].to(self.device)
        hetero_data.id = (sample['id'])

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
        hetero_data['predicate'].x = self.predicate_embeddings  # (106, 768)
        self.cache[idx] = hetero_data

        assert hetero_data['news'].num_nodes == 1, f"Expected 1 news node, got {hetero_data['news'].num_nodes}"
        assert hetero_data[
                   'predicate'].num_nodes == 106, f"Expected 106 predicate nodes, got {hetero_data['predicate'].num_nodes}"

        # 边索引
        for edge_type, edge_index in sample['edge_indices'].items():
            hetero_data[edge_type].edge_index = edge_index.to(self.device)

            # ('news', 'to_predicate', 'predicate'): torch.Size([2, 3392])
            # ('predicate', 'to_news', 'news'): torch.Size([2, 3392])
            # ('predicate', 'to_predicate_forward', 'predicate'): torch.Size([2, 178080])
            # ('predicate', 'to_predicate_backward', 'predicate'): torch.Size([2, 178080])

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import math


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
            nn.Linear(768, hidden_size),
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
            conv = HGTConv(hidden_size, hidden_size, metadata, num_heads)
            self.convs.append(nn.ModuleDict({
                'conv': conv,
                'norm': nn.ModuleDict({
                    node_type: nn.LayerNorm(hidden_size)
                    for node_type in metadata[0]
                })
            }))

        # 分类器（带初始化）
        self.classifier = Linear(hidden_size, 15)

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
        print("dynamic_pred_embs:", dynamic_pred_embs.shape)

        # Step 2: 编码新闻节点
        news_encoded = self.news_encoder(news_emb)

        # Step 3: 更新x_dict
        x_dict = {
            'news': news_encoded,
            'predicate': dynamic_pred_embs.view(-1, self.hidden_size)
        }
        print("x_dict['predicate']:", x_dict['predicate'].shape)

        # 特征转换（带残差）
        residual_dict = {k: v for k, v in x_dict.items()}
        x_dict = {k: self.lin_dict[k](v) + residual_dict[k] for k, v in x_dict.items()}

        # HGT卷积层
        attentions = []
        for layer in self.convs:
            conv = layer['conv']
            norm_dict = layer['norm']

            # 卷积操作
            x_dict, att = conv(x_dict, edge_index_dict)
            attentions.append(att)

            # 归一化 + 残差 + 激活
            x_dict = {k: F.elu(norm_dict[k](x_dict[k] + residual_dict[k]))
                      for k in x_dict}
            residual_dict = {k: v.clone() for k, v in x_dict.items()}

        # 平均注意力权重
        avg_attention = torch.mean(torch.stack(attentions), dim=0)
        # print("x_dict['news']:", x_dict['news'])
        # print("x_dict['predicate']:", x_dict['predicate'].shape)

        return self.classifier(x_dict['news']), avg_attention


def train_model(model, train_loader, val_loader):
    model = model.to(config.device)
    # opt = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
    # criterion = nn.CrossEntropyLoss()

    # 优化器配置
    optimizer = AdamW(model.parameters(),
                      lr=config.lr,
                      weight_decay=config.weight_decay)

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.epoch,
                                  eta_min=config.lr / 100)

    if config.focal_alpha == "alpha1":  # 差异最小
        alpha = [1.1135856097836927, 1.1655539811342945, 1.2189099842720732, 1.2498730000264116,
                 1.2533898762916564, 1.2797705794659595, 1.3006006695679493, 1.31009460877715,
                 1.4326544099239886, 1.4388577414488666, 1.5550319074664687, 1.5713080210486097,
                 1.6002287078920014, 1.6323716204658443, 1.6834580875735297]
    elif config.focal_alpha == "alpha2":
        alpha = [1.8807908085824854, 2.410071809351543, 3.0411733067499815, 3.449089645829311,
                 3.4973775009972528, 3.8723688846763205, 4.184427390053088, 4.331339085825812,
                 6.490084447075177, 6.612173059787246, 9.119998141410997, 9.504093898414279,
                 10.205777517506547, 11.013922293889948, 12.35783957573458]
    elif config.focal_alpha == "alpha3":
        alpha = [1.0, 1.6420220249357202, 2.614576493762311, 3.363008594565055, 3.4578331715788937,
                 4.239088233452314, 4.949839134249781, 5.303509871513632, 11.907475813544416,
                 12.359685959466862, 23.513025356026397, 25.53526970954357, 29.444976076555022,
                 34.292806484295845, 43.172193877551024]
    elif config.focal_alpha == "alpha4":  # 差异最大
        alpha = [3.537374065648359, 5.808446126231019, 9.248735081688618, 11.896219384966964,
                 12.23164938448179, 14.99524077900933, 17.5094325826265, 18.76049827640238, 42.12119613016711,
                 43.72083257257623, 83.17436609934005, 90.32780082987551, 104.15789473684211, 121.306484295846,
                 152.71619897959184]
    else:
        print("Wrong alpha")
        exit()

    gamma = config.focal_gamma

    if config.type_of_loss == "ce":
        print("\n=====config.type_of_loss =====", config.type_of_loss)
        criterion = torch.nn.CrossEntropyLoss()
    elif config.type_of_loss == "focal":
        print("\n=====config.type_of_loss =====", config.type_of_loss)
        criterion = MultiClassFocalLoss(alpha=alpha, gamma=gamma, reduction='mean')  # 这里也可以改sum!!!
    else:
        print("\nWrong name of loss")
        exit()

    # best_acc = 0

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0  # 记录连续未改善的 epoch 数
    early_stopping_patience = 500  # 设置耐心值，即最多允许 5 个 epoch 验证损失没有下降

    # val_acc, val_loss = evaluate(model, val_loader, criterion)
    # print(f"Epoch {0} 训练前val, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    for epoch in range(config.epoch):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            batch = batch.to(config.device)
            optimizer.zero_grad()
            pred, attention_probs = model(batch.x_dict, batch.edge_index_dict)
            print("pred.argmax(1):", pred.argmax(1))
            print("batch.y.squeeze():", batch.y.squeeze())
            loss = criterion(pred, batch.y.squeeze())
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        # 更新学习率
        scheduler.step()
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}")
        print("---------validating--------------")
        val_acc, val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 检查并保存最佳训练损失模型
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), config.best_train_loss_model_path)
            print(f"Epoch {epoch + 1}, Best Train Loss model saved at {config.best_train_loss_model_path}")

            # 检查并保存最佳验证损失模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.best_val_loss_model_path)
            print(f"Epoch {epoch + 1}, Best Val Loss model saved at {config.best_val_loss_model_path}")

            # 检查并保存最佳验证准确率模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.best_val_acc_model_path)
            print(f"Epoch {epoch + 1}, Best Val Acc model saved at {config.best_val_acc_model_path}")

        # print("epoch:{}的参数\n".format(epoch))
        # for name, param in model.named_parameters():
        #     if name == 'convs.2.k_rel.weight':
        #         print(f"Parameter: {name}")
        #         print(f"Shape: {param.shape}")
        #         print(f"Values: {param.data}\n")  # 直接打印值


def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            pred, attention_probs = model(batch.x_dict, batch.edge_index_dict)
            print("pred.argmax(1):", pred.argmax(1))
            print("batch.y.squeeze():", batch.y.squeeze())
            correct += (pred.argmax(1) == batch.y.squeeze()).sum().item()
            total += len(batch.y)

            loss = criterion(pred, batch.y.squeeze())
            val_loss += loss.item()
        # print("Val: correct / total:", correct / total)
    return correct / total, val_loss / len(loader)


def predict(model, loader, att_file):
    model.load_state_dict(torch.load(config.best_val_acc_model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    correct = 0
    total = 0
    graph_ids = []
    with open(att_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())  # 解析 JSON 数据
            graph_ids.append(data["id"])  # 提取 graph_id

    # with torch.no_grad():
    #     for batch in tqdm(loader):
    with torch.no_grad():
        for batch in tqdm(loader):
            batch_list = batch.id.cpu().tolist()
            # if batch_list[0] in graph_ids:
            #     print("====该id的attention已经存在！")
            #     continue
            batch = batch.to(config.device)
            pred, attention_probs = model(batch.x_dict, batch.edge_index_dict)
            # print("att:", attention_probs.shape)
            # print(attention_probs)
            correct += (pred.argmax(1) == batch.y.squeeze()).sum().item()
            print("pred.argmax(1):", pred.argmax(1))
            print("batch.y.squeeze():", batch.y.squeeze())

            total += len(batch.y)

            with open(att_file, "a") as f:  # 使用 .jsonl 扩展名，表示 JSON Lines 格式
                for i, graph_id in enumerate(batch_list):
                    # if graph_id in graph_ids:
                    #     print("====该id的attention已经存在！")
                    #     continue
                    first_node_attention = attention_probs[i, :].cpu().numpy().tolist()
                    # print("first_node_attention.len:", len(first_node_attention)) # 取第一个节点的注意力
                    json_line = json.dumps({"id": graph_id, "attention": first_node_attention})  # 转换为 JSON
                    f.write(json_line + "\n")  # 写入文件，每行一个 JSON 对象

        print("Test: correct / total:", correct / total)


def main():
    # 处理谓词嵌入
    print("# 处理谓词嵌入")
    descriptions = list(predicate.values())
    predicate_embeds = get_bert_embeddings(descriptions, config.device)
    print("predicate_embeddings.shape:", predicate_embeds.shape)

    # 加载数据
    print("# 加载数据")
    train_data = process_graph_data(config.train_data_file)
    val_data = process_graph_data(config.val_data_file)
    test_data = process_graph_data(config.test_data_file)

    # 创建数据集
    print("# 创建数据集")
    train_dataset = GraphDataset(train_data, predicate_embeds, config.device)
    val_dataset = GraphDataset(val_data, predicate_embeds, config.device)
    test_dataset = GraphDataset(test_data, predicate_embeds, config.device)

    # 获取metadata
    print("# 获取metadata")
    sample_data = train_dataset.get(0)
    metadata = (list(sample_data.x_dict.keys()),
                list(sample_data.edge_index_dict.keys()))
    print("===========metadata:\n", metadata)

    # 初始化模型
    print("# 初始化模型")
    model = HGTModel(
        predicate_bert_embs=predicate_embeds,
        hidden_size=128,
        num_heads=config.nheads,
        num_layers=3,
        metadata=metadata
    )
    # model.load_state_dict(
    #     torch.load(r'/home/s2024244178/Trust_TELLER-main/HGT_saved_models_ce_seed42/train_loss_best_model.pth',
    #                map_location=config.device))

    # model = model.to(config.device)
    # # 触发 Lazy 参数初始化：使用一个 dummy 样本前向传播
    # dummy_sample = test_dataset.get(0)  # 获取一个样本
    # dummy_sample = dummy_sample.to(config.device)
    # _ = model(dummy_sample.x_dict, dummy_sample.edge_index_dict)
    #
    # # 进行参数初始化
    # model.reset_parameters()
    # #
    # # 创建DataLoader
    print("# 创建DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # 训练和评估
    print("# 训练")
    # model.load_state_dict(torch.load(config.best_val_loss_model_path, map_location=config.device))

    train_model(model, train_loader, val_loader)
    # print(" 测试")
    # test_acc = evaluate(model, test_loader)
    # print(f'Test Accuracy: {test_acc:.4f}')

    # 预测并保存attention

    predict(model, test_loader, 'hgt_test_attentions_dynamic_evidence_tuning_temperature0.5分数不用管.jsonl')
    # predict(model, val_loader, 'hgt_train_5_attentions_dynamic_evidence_tuning_temperature0.5.jsonl')
    # predict(model, train_loader, 'hgt_train_6_attentions_dynamic_evidence_tuning_temperature0.5.jsonl')


if __name__ == '__main__':
    main()
