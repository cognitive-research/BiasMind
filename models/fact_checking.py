import argparse
import os
import re
import json
import random
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from tqdm import tqdm
from utils.prompt_loader import Init_Predicate
from utils.openaiLLM import OpenAIModel
from utils.evaluation import acc_compute, calculate_macro_f1
from utils.data_reading import read_jsonl_file
import string

Index = {"LIAR-PLUS-binary":{"MESSAGE":"statement", }}

class LogicA_Generation:
    """
    Generate general question based on claim.
    When generate answer for one general question, we adopt open-source T5-series from Google.
    We will give answer using two modes: binary and multiple classification.
    The possibility can be obtained by two modes: vocabulary, multiple sampling.
    answer model: Flan-T5, NLI Model, GPT-3.5-turbo

    See below for the size of answer models:
    Flan-T5-Small 80M
    Flan-T5-Base 250M
    Flan-T5-Large 780M
    Flan-T5-XL 3B
    Flan-T5-XXL 11B
    GPT-3.5-Turbo 200B



Init_Predicate = {'Special': {
    'VENUE': "What is the source of the information?",
    'POSTTIME': "When was it published?",
    'PUBLISHER': "Who did publish it? ",
    'REPUTATION': "How about the background and reputation of the publisher?",
    'INTENT': "What is the intent of this message?",
    'STATEMENTS': "What are the important statements of this message for verification?",
    'EVIDENCES': "What are relevant evidences to predict the veracity label of STATEMENT",
    'EVIDENCE': "What are relevant evidences to predict the veracity label of MESSAGE"
},

    """
    def __init__(self, answer_mode = 'binary', p_mode='multiple', model='Flan-T5', template=Init_Predicate, know='gold', args=None):

        self.answer_mode = answer_mode
        # p_mode: Specifies the prediction mode, supporting 'multiple' (multi-class classification)
        self.p_mode = p_mode

        self.model = model
        # initialize corresponding tokenizer
        if 'T5' in self.model:
            self.tokenizer = None
        elif '3.5' in self.model:
            self.tokenizer = None
        self.template = template
        # gold or open-book,
        self.know = know
        self.args = args
        self.index = Index[args.dataset_name]

    # Extract the question format from the template, and then generate the corresponding questions based on the input data.
    def general_question_generation(self, sample):
        # sample format id dict
        # Convert the question to a format accepted by QAmodel
        gq_dict = self.template['General']
        gq_list = gq_dict.keys()
        gq_n = len(gq_list)
        for gq_name in gq_list:
            # If GQ provides a format, follow the format; Python variables can be extracted from the format. If not, select 'G_prefix'.
            gq = gq_dict[gq_name]
            gq_format = gq[1]
            vs = self.extract_variables(gq_format)
            # save answers for all special questions
            prefix = self.execute_variables(vs, gq_format, sample)
            q_fix = self.template['Q']
            suffix = self.template['G_suffix']
            gq_format = 0
                # has a format



    def extract_variables(self, gq_format):
        pattern = r'\$(.*?)\$'
        matches = re.findall(pattern, gq_format)
        variables = []
        for match in matches:
            variables.append(match.strip())
        return variables

    # Replace the variables extracted from the template with the actual values ​​in the sample data to generate specific frequently asked questions.
    def execute_variables(self, vs: list, gq_format: str, sample: dict) -> list:
        com_gq = []
        if len(vs)>0:
            if isinstance(vs[0], list):
                # generate the i-th sub qg for one specific gq
                for i in range(len(vs[0])):
                    new_q = gq_format
                    for v in vs:
                        new_q = new_q.replace(v, sample[v][i])
                    com_gq.append(new_q)
            else:
                new_q = gq_format
                for v in vs:
                    new_q = new_q.replace(v, sample[v])
                com_gq.append(new_q)
            return com_gq
        else:
            print("Wrong Predicate with Zero Entity\n The general question is {}".format(gq_format))


    def prepare_data(self):
        if self.args.dataset_name == 'LIAR-PLUS':
            data_path = os.path.join(self.args.data_path, self.args.dataset_name)
            set_list = []
            set_paths = [os.path.join(data_path, 'train.jsonl'),
                        os.path.join(data_path, 'test.jsonl'),
                        os.path.join(data_path, 'val.jsonl')
                        ]
            for set_path in set_paths:
                set = read_jsonl_file(set_path)
                new_set = []
                for sample in set:
                    new_sample = {"ID": sample["statement id"],
                                  "MESSAGE":sample["statement"],
                                  "EVIDENCE":sample["evidence"],
                                  'VENUE': sample["the context"],
                                  'POSTTIME': None,
                                  }

                    # new_sample[]

        pass






def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='LIAR-PLUS', type=str)
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--mode', type=str, default='multiple', choices=['binary', 'multiple'])
    # choose fewer smale for testing
    parser.add_argument('--num_eval_samples', default=5, type=int)
    parser.add_argument('--shot_number', default=0, type=int)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--save_all_path', default='../data/report.jsonl', type=str)
    parser.add_argument('--run_number', default=1, type=int)
    parser.add_argument('--know', default=None, type=str, choices=[None, 'gold', 'wiki', 'google'])

    parser.add_argument('--api_key', type=str, default='default')
    parser.add_argument('--model_name', type=str, default='text-davinci-003', choices=['text-davinci-003', 'gpt-3.5-turbo'])
    parser.add_argument('--stop_words', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--logprobs', type=int, default=5)
    parser.add_argument('--echo',  action="store_true")
    parser.add_argument('--cl', action="store_true")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    # seed_everything(1024)
    # # args.mode = 'multiple'
    # # args.know = 'gold'
    # # args.model_name = 'gpt-3.5-turbo'
    # algorithm = ZeroshotCoT(args)
    # algorithm.run()