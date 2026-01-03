import argparse
import torch
import os
import numpy as np

os.environ['HF_HOME'] = "/path/code/BiasMind/"

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import snapshot_download
from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.openaiLLM import OpenAIModel
from utils.data_reading import read_openapi
# from mydeepseek import MyDS  # 调用ds api
from .mydeepseek import MyDS  # 调用ds api
# from myds_local import MyDS
# # 调用本地ds
from .mygpt import MyGPT
# from mygpt import MyGPT

"""
@misc{https://doi.org/10.48550/arxiv.2210.11416,
  doi = {10.48550/ARXIV.2210.11416},
  
  url = {https://arxiv.org/abs/2210.11416},
  
  author = {Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason},
  
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Scaling Instruction-Finetuned Language Models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
"""
"""
need set max_length for longer text 
"""

FT5_VARIANT = ["flan-t5-xxl", "flan-t5-xl", "flan-t5-large", "flan-t5-base", "flan-t5-small"]
LLAMA2_VARIANT = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf"]

DEEPSEEK_VARIANT = ["DeepSeek-V3"] #"DeepSeek-V3"表示调用本地model,
GPT_VARIANT = ["gpt-3.5-turbo"]

DEEPSEEK_API_VARIANT = ["deepseek-chat", "deepseek-reasoner"] #   The deepseek-chat model points to DeepSeek-V3. The deepseek-reasoner model points to DeepSeek-R1

DEEPSEEK_PATH = "/path/code/Pre_model/" # 记得改
FT5_PATH = "/path/code/Pre_model/"
Llama_PATH = "/path/code/Pre_model/"


YES_TOKEN_ID = 19739
NO_TOKEN_ID = 4168


class T5_Question_Answering:
    def __init__(self, model_name: str = "flan-t5-large", device: str = "cpu"):
        self.model_name = model_name  # default deepseek-chat
        self.device = device
        if model_name in FT5_VARIANT:
            path = os.path.join(FT5_PATH, model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(path)
            if device != "cpu":
                device_map = "auto"
            else:
                device_map = "cpu"
            self.model = T5ForConditionalGeneration.from_pretrained(path, device_map=device_map)
            GenerationConfig.from_pretrained(path)
            # pre-define token ids of yes and no.
            self.yes_token_id = 2163  # self.tokenizer.get_vocab()["Yes"]
            self.no_token_id = 465  # self.tokenizer.get_vocab()["No"]

        elif model_name in DEEPSEEK_VARIANT:
            path = os.path.join(DEEPSEEK_PATH, model_name)
            if device != "cpu":
                device_map = "auto"
            else:
                device_map = "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(path)

            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map=device_map,
                torch_dtype = torch.float16,
                trust_remote_code=True
            )

            self.yes_token_id = self.tokenizer.get_vocab()["是"]
            self.no_token_id = self.tokenizer.get_vocab()["否"]


        elif model_name in LLAMA2_VARIANT:
            path = os.path.join(Llama_PATH, model_name)
            if device != "cpu":
                device_map = "auto"
            else:
                device_map = "cpu"
            # print("============开始创建tokenizer===============")
            self.tokenizer = LlamaTokenizer.from_pretrained(path)

            # print("============开始创建llama模型===============")
            # self.model = LlamaForCausalLM.from_pretrained(path, device_map=device_map, torch_dtype=torch.bfloat16)
            self.model = LlamaForCausalLM.from_pretrained(path, device_map=device_map, torch_dtype=torch.float16)


            # pre-define token ids of yes and no.
            self.yes_token_id = self.tokenizer.get_vocab()["Yes"]  # 8241
            self.no_token_id = self.tokenizer.get_vocab()["No"]  # 3782


        elif model_name in GPT_VARIANT:
            self.model = MyGPT(model_name=self.model_name)
            # self.model = OpenAIModel(API_KEY=read_openapi(), model_name=self.model_name, stop_words=[],
            #                          max_new_tokens=1, logprobs=None, echo=False)


        elif model_name in DEEPSEEK_API_VARIANT: # deepseek_v3
            self.model = MyDS(model_name=self.model_name)

        else:
            print("Wrong model version {} ".format(self.model_name))

    # 下面的answer_direct_sampling会调用这个 ！  调用模型生成文本答案， 报错 string indices must be integers
    def generate(self, input_string, **generator_args):
        #  Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary. not for batch operation
        if self.model_name in FT5_VARIANT or self.model_name in LLAMA2_VARIANT:
            input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.model.generate(input_ids, **generator_args)
            #     将 ID 序列转换回文本，并去掉特殊符号
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)
        else:
            # 获取生成的多个回答，并提取 message["content"] 作为最终答案，是一个batch 送入的吗，感觉不是，是一个一个生成的
            res = self.model.batch_chat([input_string], temperature=generator_args['temperature'],
                                                 n=generator_args['num_return_sequences'])
            # 注意调deepseek n=1就可，因为会取概率
            # res 是包含 n个Yes/No的列表,这里 n次回答,n =3时候 ['Yes', 'No', 'Yes']
            # print("res in generate!!!",res)  # 对了
            return res

    # 对于多分类 得到分数和"P7 "的映射
    def answer_multiclas_sampling(self, message):
        answer_score_dic =  self.model.batch_chat_multiclas([message], temperature=1.0,
                                                 n=1)
        return answer_score_dic


    # 对于大模型生成的谓词 + 43其他宣传手段，每个都调用一次answer_direct_samplin得到对应的分数，如何转分类
    # deepseek!  使用采样的方法 生成多个答案，并计算 "Yes" 的可能性
    def answer_direct_sampling(self, info, gq, do_sample=True, temperature=1, num_return_sequences=1):  # 这里改温度， 改num
        # print("============answer_direct_sampling....===============")
        # info 是message, gq是谓词问题
        # deepseek 中文prompt
        if info is None:
            input_string = ["{} 是或否? 请回答::".format(gq)]
        else:

            input_string = ["{}\n. {} 是或否? 请回答:".format(info, gq),
                       "{}\n 基于以上信息, {} 是或否? 请回答:".format(info, gq)]


        # if info is None:
        #     input_string = "{}\n Please response Yes or No:".format(gq)
        # else:
        #     input_string = "{}\n {} Please response Yes or No:".format(info, gq)
        #     print("===in else now prompt:{}".format(input_string))

        # do_sample=True 允许模型进行采样，生成不同的答案
        # 调用 generate() 生成 num_return_sequences 个答案
        # max_new_tokens=3 限制生成长度

        answer_texts = self.generate(input_string,
                                     max_new_tokens=1, do_sample=do_sample, temperature=temperature,
                                     num_return_sequences=num_return_sequences)

        # print("===now answer_texts:{}".format(answer_texts))
        # 如果模型是 Flan-T5，可能会重复 input_string，所以去掉 input_string 只保留答案部分
        if self.model_name in FT5_VARIANT or self.model_name in LLAMA2_VARIANT:
            # answer_texts = [a[len(input_string):] for a in answer_texts if len(a) > len(input_string)]
            answer_texts = [a[len(input_string):].strip().split()[0] for a in answer_texts if
                            len(a) > len(input_string)]
            print("保留答案部分：{}".format(answer_texts))

        # 如果调用ds api，可以直接得到分数，answer_texts 即为分数
        if self.model_name in DEEPSEEK_API_VARIANT:
            f_score =  answer_texts
        else:
            # 把 Yes/No 变成 1/0
            answer_texts = self.map_direct_answer_to_label(answer_texts)

            # 计算 Yes 的比例
            l = len(answer_texts)
            if l > 0:
                f_score = sum(answer_texts) / l
            else:
                # or rewrite to 0.5
                f_score = None
        # here, f_score is the possibility that input_string is yes
        # print("=======f_score: {}".format(f_score))
        return f_score

    # 将 "Yes"/"No" 文本转换为数值标签 (Yes=1, No=0)
    def map_direct_answer_to_label(self, predicts):
        predicts = [p.lower().strip() for p in predicts]
        # print("***********predicts:{}".format(predicts)) # ['no', 'yes', 'no']
        label_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'YES': 1, 'NO': 0}  #考虑了不同的yes 和no
        labels = label_map.keys()
        results = [label_map[p] for p in predicts if p in labels]
        print("***********results:{}".format(results)) # [0, 1, 0]
        return results


    # info是证据，gq是问题
    def answer_logics(self, info, gq, **kwargs) -> float:
        # print("============answer_logics....===============")
        # return the possibility that the answer to gq is Yes
        scores = []
        # Multiple prompts for robustness
        if self.model_name in FT5_VARIANT:
            if info is None:
                prompts = ["{} Yes or No? Response:".format(gq)]
                # to process overlong gq
            else:

                prompts = ["{} {} Yes or No? Response:".format(info, gq),
                           "{}\nBased on the above information, {} Yes or No? Response:".format(info, gq),
                           ]
            for input_string in prompts:
                print("===now prompt:{}".format(input_string))
                input_id = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)

                output = self.model.generate(
                    input_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1,
                )
                v_yes_exp = (
                    torch.exp(output.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
                )
                v_no_exp = (
                    torch.exp(output.scores[0][:, self.no_token_id]).cpu().numpy()[0]
                )

                score = v_yes_exp / (v_yes_exp + v_no_exp)

                print("===v_yes_exp:{}".format(v_yes_exp))
                print("===v_no_exp:{}".format(v_no_exp))
                print("===score:{}".format(score))
                scores.append(score)
            f_score = float(np.mean(scores))
            print("===final score:{}".format(f_score))
            return f_score
        elif self.model_name in LLAMA2_VARIANT:
            if info is None:
                prompts = ["{} Yes or No? Response:".format(gq)]
            else:
                # tmp = self.tokenizer.encode(info)
                # if len(tmp) > 500:
                #     info = self.tokenizer.decode(tmp[:500])
                prompts = ["{}\n. {} Yes or No? Response:".format(info, gq),
                           "{}\n Based on the above information, {} Yes or No? Response:".format(info, gq)]
            for input_string in prompts:
                print("===now prompt:{}".format(input_string))
                input_id = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    predictions = self.model(input_id)[0]
                    next_token_candidates_tensor = predictions[0, -1, :]

                    v_yes_exp = (
                        next_token_candidates_tensor[self.yes_token_id].float().cpu().numpy()
                    )
                    v_no_exp = (
                        next_token_candidates_tensor[self.no_token_id].float().cpu().numpy()
                    )

                    score = v_yes_exp / (v_yes_exp + v_no_exp)

                    print("===v_yes_exp:{}".format(v_yes_exp))
                    print("===v_no_exp:{}".format(v_no_exp))
                    print("===score:{}".format(score))
                    scores.append(score)
            f_score = float(np.mean(scores))
            print("===final score:{}".format(f_score))
            return f_score

        elif self.model_name in DEEPSEEK_VARIANT:
            if info is None:
                prompts = ["{} 是或否? 请回答:".format(gq)]
            else:
                # tmp = self.tokenizer.encode(info)
                # if len(tmp) > 500:
                #     info = self.tokenizer.decode(tmp[:500])
                prompts = ["{}\n. {} 是或否? 请回答:".format(info, gq),
                           "{}\n 基于以上信息, {} 是或否? 请回答:".format(info, gq)]
            for input_string in prompts:

                input_id = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
                output = self.model.generate(
                    input_id,
                    max_new_tokens=50,
                    output_scores=True,  # 启用 logits 输出
                    return_dict_in_generate=True
                )
                v_yes_exp = (
                    torch.exp(output.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
                )
                v_no_exp = (
                    torch.exp(output.scores[0][:, self.no_token_id]).cpu().numpy()[0]
                )   # 再gai

                score = v_yes_exp / (v_yes_exp + v_no_exp)

                scores.append(score)
            f_score = float(np.mean(scores))
            # print("===final score:{}".format(f_score))
            return f_score
        else:
            print("Model {} can not use logics to decide the truth value of predicates. ".format(self.model_name))
            exit()





def download_T5(path=FT5_PATH):
    # download five variants of Flan-T5 to FT5_PATH in server
    # "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
    # "google/flan-t5-xl", "google/flan-t5-xxl",
    model_name_list = ["google/flan-t5-xl", "google/flan-t5-xxl"]
    for model_name in model_name_list:
        mn = model_name.split("/")[-1]
        dir = os.path.join(path, mn)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # snapshot_download(repo_id=model_name, local_dir=dir)
        snapshot_download(repo_id=model_name, local_dir=dir, ignore_patterns=["*.h5", "*.msgpack"])


def download_llama2(path=Llama_PATH):
    # download five variants of Flan-T5 to FT5_PATH in server
    # "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
    # "google/flan-t5-xl", "google/flan-t5-xxl",
    # "meta-llama/Llama-2-13b-hf"
    # "meta-llama/Llama-2-13b-chat-hf"
    model_name_list = ["meta-llama/Llama-2-7b-chat-hf"]
    for model_name in model_name_list:
        mn = model_name.split("/")[-1]
        dir = os.path.join(path, mn)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # set your own tokens
        snapshot_download(repo_id=model_name, local_dir=dir, ignore_patterns=["*.h5", "*.msgpack"], token="")


if __name__ == "__main__":

    qa = T5_Question_Answering(model_name="Llama-2-7b-chat-hf")
    q = "Is Trump better than Biden?"
    a = qa.answer_logics(info=None, gq=q)
    b = qa.answer_direct_sampling(info=None, gq=q)
    print(a)
    print(b)
