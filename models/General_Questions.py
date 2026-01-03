import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import random
import sys
import time

# python General_Questions.py

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from utils.prompt_loader import Init_Predicate, Evolving_Predicate
import re
from tqdm import tqdm
from qa_t5 import T5_Question_Answering
from utils.data_reading import whole_data_load_GQ, test_data_load_GQ, val_data_load_GQ, write_json_file, read_json_file

# Sleep for the specified delay

# Use a large model to generate instanced scores
# Call the delayed function
"""
This class aims to produce the answer for general questions.
"""


class GeneralQuestion:
    def __init__(self, qa_model_name="flan-t5-large", device="cpu", mode="logics", task="binary", evo_flag=False):
        self.device = device
        self.qa_model_name = qa_model_name
        self.qa_model = T5_Question_Answering(model_name=self.qa_model_name, device=self.device)
        self.qa_mode = mode
        self.evo_flag = evo_flag
        if self.evo_flag:
            self.gq = Evolving_Predicate['General']
        else:
            self.gq = Init_Predicate['General']
        self.gq_list = self.gq.keys()
        self.gq_n = len(self.gq) # Number of questions
        self.task = task
        print("********* Predefined General Questions *********\n")
        for key in self.gq_list:
            print(self.gq[key])

    # save_path是gq_file_path data/dataset/ xx.json
    def run(self, dataset_name, data_path, sq_file_path, openbook, save_path):
        # generate logic atoms for one dataset
        # datasest = whole_data_load_GQ(data_path=data_path, dataset_name=dataset_name,
        #                               sq_file_path=sq_file_path, openbook=openbook)


        datasest = test_data_load_GQ(data_path=data_path, dataset_name=dataset_name,
                                      sq_file_path=sq_file_path, openbook=openbook)

        # datasest = val_data_load_GQ(data_path=data_path, dataset_name=dataset_name,
        #                               sq_file_path=sq_file_path, openbook=openbook)

        #   sq_file_path = os.path.join(data_path, "sq.json")  sq.json {"ID 4258": {"STATEMENTS": [s1,s2,s3]}}

        attempt = 1

        retries = 5  # def 100
        # fix some exceptions to gpt
        while attempt <= retries:
            try:
                res = self.gen_ans_samples(datasest, save_path)
                print("res:",res) # id,key, value {P1:[[info1, [instance1, Probability1],[info2, [instance2, Probability2]], P2:...}
                #  info (instance_name + ":" + instance)
                write_json_file(res, save_path)
                print("The file has saved to {}".format(save_path))
                return res
            except Exception as e:
                print(f"Exception occurred: {str(e)}")
                print(f"Retrying... (Attempt {attempt}/{retries})")
                attempt += 1
                time.sleep(5)

    def gen_ans_samples(self, samples, save_path):
        results = {}
        if os.path.exists(save_path):
            results = read_json_file(save_path)
        try:
            for idx in tqdm(samples.keys()):
                if str(idx) not in results.keys():
                    results[str(idx)] = self.gen_ans_one_sample(samples[idx])
                    write_json_file(results, save_path)

                else:
                    continue
        except Exception as e:
            print(e)
            write_json_file(results, save_path)
            raise
        return results


    def gen_ans_one_sample(self, sample):
        # input format {P1:[[info, [instance1, instance2]]], P2:...}

        # print("=======sample:=====",sample["MESSAGE"])


        gq_dic = self.gq_gen(sample)
        # print("=====gq_dic:",gq_dic)
        # e.g. gq_name: "P1"; gqs: [info, n_inst]

        # Define the mapping relationship between existing promotional methods and predicate names in the dataset.
        dataset_propaganda_mapping = {
            'P1': '加载语言（情绪语言）',
            'P2': '贴标签',
            'P3': '挥舞旗帜（高举大旗）',
            'P4': '简化因果',
            'P5': '喊口号',
            'P6': '非黑即白',
            'P8': '诉诸反复',
            'P10': '诉诸权威',
            'P11': '诉诸恐惧/偏见',
            'P12': '格言论证',
            'P13': '诉诸转移',
            'P14': '引用历史',
            'P15': '预设立场',
            'P16': '诉诸潮流',
            'P34': '夸张（夸大或淡化）',
            'P36': '诉诸质疑'
        }


        else_propaganda_mapping = {
            'P7': '诉诸人身',
            'P9': '议题设置',
            'P17': '美好人生',
            'P18': '戈培尔效应',
            'P19': '假二难推理',
            'P20': '单方论证',
            'P21': '经典条件反射',
            'P22': '认知失调',
            'P23': '诉诸平民',
            'P24': '个人崇拜',
            'P25': '铁魔化',
            'P26': '低落士气',
            'P27': '独裁',
            'P28': '虚假信息',
            'P29': '分而治之',
            'P30': '以退为进',
            'P31': '粗直语',
            'P32': '委婉语',
            'P33': '制造幸福感',
            'P35': '虚假指控',
            'P37': '谎言灌喷',
            'P38': '得寸进尺',
            'P39': '框架化',
            'P40': '煤气灯效应',
            'P41': '乱枪打鸟',
            'P42': '光辉普照',
            'P43': '关联谬误',
            'P44': '片面事实',
            'P45': '含糊其辞',
            'P46': '接受范围',
            'P47': '负载性语言',
            'P48': '谎言与欺骗',
            'P49': '社会环境控制',
            'P50': '冷处理',
            'P51': '人身攻击',
            'P52': '形式谬误',
            'P53': '混淆',
            'P54': '操作制约',
            'P55': '单音谬误',
            'P56': '单一思想',
            'P57': '断章取义',
            'P58': '暗示性语言',
            'P59': '终结性陈词',
        }

        frames_mapping = {
            'P102': '损失与获益框架',
            'P103': '属性框架',
            'P104': '时间框架',
            'P105': '行动框架',
            'P106': '社会比较框架',

        }
        structures_mapping = {
            'P94': '问题-解决结构',
            'P95': '对比结构',
            'P96': '英雄之旅结构',
            'P97': '三幕剧结构',
            'P98': '框架叙事结构',
            'P99': '循环叙事结构',
            'P100': '嵌套结构',
            'P101': '并行结构',
        }


        # {'P9': 0.9999997615814785, 'P44': 0.999808490280328}
        if self.qa_mode == 'sampling':
            multiclas_dic = self.qa_model.answer_multiclas_sampling(sample["MESSAGE"])


        for gq_name, gqs in gq_dic.items():


            propaganda_name = dataset_propaganda_mapping.get(gq_name)
            # print("=====propaganda_name:",propaganda_name)

            structures = structures_mapping.get(gq_name)
            frames = frames_mapping.get(gq_name)
            else_propaganda_name = else_propaganda_mapping.get(gq_name)

            if propaganda_name:
                # print("---------------------------------propaganda_name:", propaganda_name)
                for gq in gqs:

                    if sample['PROPAGANDA'] == propaganda_name:
                        s = 1.0
                    else:
                        s = 0.0

                    gq.append(s)

            elif structures:
                # print("-------------------------------------structures:", structures)
                for gq in gqs:

                    if structures in sample['STRUCTURES']:
                        # print("--------------------------sample['STRUCTURES'][0]:", sample['STRUCTURES'][0])
                        s = 1.0
                    else:
                        # print("--------------------------structures---000---'")
                        s = 0.0
                    gq.append(s)
                    # print("gq_structures")

            elif frames:
                # print("---------------------------------frames:", frames)
                for gq in gqs:

                    if frames in sample['FRAME']:
                        # print("-------------------------sample['FRAME'][0]:", sample['FRAME'][0])
                        s = 1.0
                    else:
                        s = 0.0
                    gq.append(s)
                    # print("gq_frames")

            elif else_propaganda_name:
                # print("===in else_propaganda_name======",else_propaganda_name)
                for gq in gqs:
                    if self.qa_mode == 'logics':
                        s = self.qa_model.answer_logics(gq[0], self.gq[gq_name][0])  # , info, gq,
                    else:
                        # print("------gq[0]-----", gq[0])  # Message:
                        # print("------gq_name-----", gq_name)  # P7
                        s = multiclas_dic[gq_name]
                        # print("===s===",s)
                    gq.append(s)

            else:
                # print("===llm generated predicate======")
                for gq in gqs:
                    if self.qa_mode == 'logics':

                        s = self.qa_model.answer_logics(gq[0], self.gq[gq_name][0])  # , info, gq,
                    else:

                        # print("------self.gq[gq_name][0]-----",self.gq[gq_name][0])  # Does the message use phrases lik(P1)
                        s = self.qa_model.answer_direct_sampling(gq[0], self.gq[gq_name][0])

                    gq.append(s)
        # output format  {P1:[[info, [instance1, instance2], Probability that logic atom is true]], P2:...}
        # print("=========gq_dic",gq_dic)
        return gq_dic


    def gq_gen(self, sample):


        '''
        'General': {
        # each sample in dataset is a dict, EVIDENCE/Message (the second element of the tuple) is the key of the dict.
        'P1': ["Is the message true?", [("Background information", "EVIDENCE"), ("Message", "MESSAGE")]],
        'P2': ["Did the message contain adequate background information?", [("Message", "MESSAGE")]],
        'P3': ["Is the background information in message accurate and objective?", [("Message", "MESSAGE")]],
        'P4': [
            "Is there any content in message that has been intentionally eliminated with the meaning being distorted?",
            [("Message", "MESSAGE")]],
        'P5': ["Is there an improper intention (political motive, commercial purpose, etc.) in the message?",
               [("Message", "MESSAGE"), ("Intent", "INTENT")]],
        'P6': ["Does the publisher have a history of publishing information with improper intention?",
               [("Publisher Reputation", "REPUTATION")]],
        'P7': ["Is the statement true?", [("Background information", "EVIDENCES"), ("Statement", "STATEMENTS")]],
        'P8': ["Is the message false?", [("Background information", "EVIDENCE"), ("Message", "MESSAGE")]],
    },
        '''
        gq_dic = {}
        for gq_name, gq in self.gq.items():  #  self.gq = Init_Predicate['General']
            # gq_name : 'P1'
            # gq: ["Is the message true?", [("Background information", "EVIDENCE"), ("Message", "MESSAGE")]],

            gq_formats = gq[1]
            gq_dic[gq_name] = []
            l = len(gq_formats)
            # in this case, the gq corresponds to n atoms


            if isinstance(sample[gq_formats[0][1]], list):

                n = len(sample[gq_formats[0][1]])
                # the i-th atoms for the gq
                for i in range(n):
                    # the j-th instance for the i-th atom'
                    valid_flag = 0
                    info = ""
                    n_inst = []
                    for j in range(l):
                        instance_name = gq_formats[j][0]
                        instance = sample[gq_formats[j][1]][i]
                        if instance is not None:
                            # the description of sample[gq_formats[j][1]]
                            info = info + instance_name + ":" + instance + "\n"
                            n_inst.append(instance)
                            valid_flag += 1
                    if valid_flag > 0:
                        gq_dic[gq_name].append([info, n_inst]) #   # gq_name : 'P1'
            # in this case, the gq corresponds to 1 atom
            else:
                info = ""
                n_inst = []
                # the j-th instance for the atom
                valid_flag = 0
                for j in range(l):
                    instance_name = gq_formats[j][0]
                    instance = sample[gq_formats[j][1]]
                    if instance is not None:
                        info = info + instance_name + ":" + instance + "\n"
                        n_inst.append(instance)
                        valid_flag += 1
                if valid_flag > 0:
                    gq_dic[gq_name].append([info, n_inst])

        # 'P94': [], 'P95': [], 'P96': [], 'P97: [], 'P101': [], 'P102': [], 'P103': [], 'P104': [], 'P105': [], 'P106': []}
        return gq_dic


    def update_gq(self, gqs):
        new_gq = {}
        for i, gq in enumerate(gqs):
            ii = i + 1 + self.gq_n
            k = "P" + str(ii)
            new_gq[k] = gq
        self.gq.update(new_gq)
        self.gq_list = self.gq.keys()
        self.gq_n = len(self.gq)

    def find_new_gqs(self):
        # search new gq by gpt-3.5
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='cognitive', type=str,
                        choices=["Constraint", "GOSSIPCOP", "LIAR-PLUS", "POLITIFACT", "cognitive"])
    parser.add_argument('--data_path', type=str, default='/path/code/BiasMind/data')
    # choose fewer smale for testing

    parser.add_argument('--model_name', type=str, default="deepseek-chat",
                        choices=["deepseek-reasoner", "flan-t5-xxl", "flan-t5-xl", "flan-t5-large", "flan-t5-base", "flan-t5-small",
                                 "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "gpt-3.5-turbo", "DeepSeek-V3"])  # The deepseek-chat model points to DeepSeek-V3. The deepseek-reasoner model points to DeepSeek-R1
    parser.add_argument('--mode', type=str, default="sampling", choices=["logics", "sampling"])
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--openbook', type=str, default="False")
    parser.add_argument('--sq_file_path', default=None, type=str)
    parser.add_argument('--gq_file_path', default=None, type=str)
    parser.add_argument('--evo_flag', action="store_true")
    parser.add_argument('--task', type=str, default="binary", choices=["binary", "multiple"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("相关args如下：\n")
    print(args)
    # # args.evo_flag = True
    # # args.openbook = "True"
    GQ = GeneralQuestion(qa_model_name=args.model_name, device=args.device, mode=args.mode, task=args.task,
                         evo_flag=args.evo_flag)
    data_path = os.path.join(args.data_path, args.dataset_name)
    print("data_path:", data_path)

    if args.sq_file_path is None:
        args.sq_file_path = os.path.join(data_path, "sq.json")

    if args.gq_file_path is None:

        if GQ.evo_flag:
            args.gq_file_path = os.path.join(data_path, args.model_name + "_" + args.openbook + "_" + "evo" + ".json")
        else:
            args.gq_file_path = os.path.join(data_path, args.model_name + "_" + args.openbook + "_multi_cla" + "_test" + ".json")
    print("sq_file_path: ", args.sq_file_path)
    print("gq_file_path: ", args.gq_file_path)
    print("================== GQ.run ================")
    GQ.run(data_path=data_path, dataset_name=args.dataset_name,
           sq_file_path=args.sq_file_path, openbook=args.openbook, save_path=args.gq_file_path)


