import os
import re

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils.data_reading import load_data_for_expert, read_yaml_file, read_json_file, write_json_file, initialize_bert_model
from models.qa_t5 import T5_Question_Answering, FT5_VARIANT, GPT_VARIANT, LLAMA2_VARIANT
import argparse
from utils.evaluation import acc_compute, calculate_macro_f1
from tqdm import tqdm
from utils.components.dnf_layer import LogicTrainer
import torch
import random
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from utils.components.dnf_layer import batch_generation, transform_org_to_logic
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score


torch.autograd.set_detect_anomaly(True)

def batch_iter(configure, s_set, gq, mask_flag, mode, batchsize, cog_name='结果偏差'):
    text_inputs, logics_input, label_input = transform_org_to_logic(configure, s_set, gq,
                                                       mask_flag=mask_flag)
    # print("=============logics_input, label_input:", logics_input, label_input)
    loader = batch_generation(text_inputs, logics_input, label_input, mode, batchsize, cog_name)
    return loader


class Expert:
    def __init__(self, dataset_name, mode, data_path, gq_file, sq_file, model_name, args):
        self.dataset_name = dataset_name
        # label rule, choice = {"binary", "multiple"}
        self.mode = mode
        self.evo_flag = args.evo_flag
        self.data_path = os.path.join(data_path, self.dataset_name)
        self.gq_file = gq_file
        self.sq_file = sq_file
        self.evo_file = args.evo_file
        self.model_name = model_name
        self.args = args
        

        if args.type_of_logic_model == "hgt_logic":
            device = args.device if torch.cuda.is_available() else 'cpu'
            print(f"Initializing BERT model for HGT_DNF on device: {device}")
            initialize_bert_model(device)
        
        self.dataset, self.rule = load_data_for_expert(data_path=self.data_path, dataset_name=self.dataset_name,
                                                       mode=self.mode, gq_file=self.gq_file, sq_file=self.sq_file,
                                                       evo_file=self.evo_file, evo_flag=self.evo_flag, cog_name=args.cog_name)

        self.save_path = args.save_path

        self.trainer = None


    def train_logic(self, num_conjuncts, n_out, configure, weight_init_type, args, exp=None):
        predicate_set = {}
        for a in configure:
            predicate_set[a[0]] = a[1]
        # configure = [('P1', 1), ('P2', 1), ('P3', 1), ('P4', 1), ('P5', 1),  ('P6', 1), ('P7', 3)]d
        # prepare train, val, test datasets
        train_set = self.dataset["train"]  #          new_set.append({"ID": ID, "MESSAGE": MESSAGE, "EVIDENCE": EVIDENCE, "label": label})
        random.shuffle(train_set)
        val_set = self.dataset["val"]
        test_set = self.dataset["test"]
        gq = self.dataset["gq"]



        train_text_inputs, train_logics_inputs, train_label_inputs = transform_org_to_logic(configure, train_set, gq,
                                                                         mask_flag=args.mask_flag)
        train_set = [train_text_inputs, train_logics_inputs, train_label_inputs]


        val_loader = batch_iter(configure, val_set, gq, mask_flag=args.mask_flag, mode=self.mode,
                                batchsize=args.batchsize, cog_name=args.cog_name)
        test_loader = batch_iter(configure, test_set, gq, mask_flag=args.mask_flag, mode=self.mode,
                                 batchsize=args.batchsize, cog_name=args.cog_name)

        print("length of train_loader {}, length of val_loader  {}, length of test_loader {}".format(
            len(train_set[0]) // args.batchsize, len(val_loader), len(test_loader)))
        args.n_steps_per_epoch = len(train_set[0]) // args.batchsize

        # initialize the training class
        if args.type_of_logic_model == "tree":
            clf = DecisionTreeClassifier(random_state=0, max_depth=5, max_leaf_nodes=10, min_weight_fraction_leaf=0.01)
            # clf = GaussianNB()
            ind_list = [i for i in range(len(train_set[0]))]
            random.shuffle(ind_list)
            # may be change tos shuffle per epoch
            train_logics_inputs = [train_set[0][i] for i in ind_list]
            train_label_inputs = [train_set[1][i] for i in ind_list]

            train_loader = batch_generation(train_logics_inputs, train_label_inputs, self.mode, args.batchsize, args.cog_name)

            train_data = torch.cat([tmp[0] for tmp in train_loader], dim=0).numpy()
            train_label = torch.cat([tmp[1] for tmp in train_loader]).numpy()
            clf.fit(train_data, train_label)
            t_data = torch.cat([tmp[0] for tmp in test_loader], dim=0).numpy()
            t_label = torch.cat([tmp[1] for tmp in test_loader]).numpy()
            p_label = clf.predict(t_data)
            # Compute accuracy
            accuracy = accuracy_score(t_label, p_label)

            # Compute macro-F1 score
            macro_f1 = f1_score(t_label, p_label, average='macro')
            # plt.figure(dpi=500)
            # tree.plot_tree(clf)
            # plt.show()
            print(accuracy, macro_f1)
            return accuracy
        if args.type_of_logic_model == "bayes":
            clf = GaussianNB()
            ind_list = [i for i in range(len(train_set[0]))]
            random.shuffle(ind_list)
            # may be change tos shuffle per epoch
            train_logics_inputs = [train_set[0][i] for i in ind_list]
            train_label_inputs = [train_set[1][i] for i in ind_list]

            train_loader = batch_generation(train_logics_inputs, train_label_inputs, self.mode, args.batchsize, args.cog_name)

            train_data = torch.cat([tmp[0] for tmp in train_loader], dim=0).numpy()
            train_label = torch.cat([tmp[1] for tmp in train_loader]).numpy()
            clf.fit(train_data, train_label)
            t_data = torch.cat([tmp[0] for tmp in test_loader], dim=0).numpy()
            t_label = torch.cat([tmp[1] for tmp in test_loader]).numpy()
            p_label = clf.predict(t_data)
            # Compute accuracy
            accuracy = accuracy_score(t_label, p_label)

            # Compute macro-F1 score
            macro_f1 = f1_score(t_label, p_label, average='macro')
            # plt.figure(dpi=500)
            # tree.plot_tree(clf)
            # plt.show()
            print(accuracy, macro_f1)
            return accuracy

        else:
            # train the logic model！
            trainer = LogicTrainer(num_conjuncts=num_conjuncts, n_out=n_out, delta=args.initial_delta,
                                   configure=configure,
                                   weight_init_type=weight_init_type, device=self.args.device, args=args, exp=exp)
            reported_test_metrics = trainer.train(train_set, val_loader, test_loader)

            return reported_test_metrics


    def eval_gq(self, model_name, device, evi_flag: bool, mode: str):
        assert (
                model_name in FT5_VARIANT or model_name in LLAMA2_VARIANT or model_name in GPT_VARIANT), "wrong model name for flan-t5 or Llama2"
        print(device)
        eval_model = T5_Question_Answering(model_name=model_name, device=device)
        test_set = self.dataset["test"]
        label_set = list(set(self.rule.values()))
        # define the gq
        real_label_set = []
        predicted_label_set = []
        for sample in tqdm(test_set):
            label_score = []
            real_label_set.append(sample["label"])
            for label in label_set:
                gq = "Message: {}\nIs the message is {}?".format(sample["MESSAGE"], label)
                if mode == 'logics':
                    if evi_flag:
                        label_score.append(eval_model.answer_logics(info=sample["EVIDENCE"], gq=gq))
                    else:
                        label_score.append(eval_model.answer_logics(info=None, gq=gq))
                else:
                    if evi_flag:
                        label_score.append(eval_model.answer_direct_sampling(info=sample["EVIDENCE"], gq=gq))
                    else:
                        label_score.append(eval_model.answer_direct_sampling(info=None, gq=gq))
            maximum = max(label_score)
            max_index = label_score.index(maximum)
            predicted_label_set.append(label_set[max_index])
        acc = acc_compute(predicted_label_set, real_label_set)
        macro_f1, macro_precision, macro_recall = calculate_macro_f1(predicted_label_set, real_label_set)
        prefix = "\ndataset_name: {}, model_name: {}, evi_flag: {}, mode: {}, class: {}".format(self.dataset_name,
                                                                                              model_name, evi_flag,
                                                                                              mode, self.mode)
        res = "\nacc: {:4f}, macro_f1: {:4f}, macro_precision: {:4f}, macro_recall: {:4f}".format(acc, macro_f1,
                                                                                                macro_precision,
                                                                                                macro_recall)
        print(prefix)
        print(res)
        reported_res = {}
        # return the accuracy/macro-f1
        if not os.path.exists(self.save_path):
            reported_res[prefix] = [res]
        else:
            reported_res = read_json_file(self.save_path)
            if reported_res is None:
                reported_res[prefix] = [res]
            if prefix in reported_res.keys():
                reported_res[prefix].append(res)
            else:
                reported_res[prefix] = res

        write_json_file(reported_res, self.save_path)
    # def eval_logic(self):
    #     # initialize the logic module


def parse_args():
    parser = argparse.ArgumentParser()


    # proser args new add
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='Proser', type=str, help='Recognition Method')
    # parser.add_argument('--backbone', default='WideResnet', type=str, help='Backbone type.')

    parser.add_argument('--known_class', default=2, type=int, help='number of known class')
    parser.add_argument('--seed', default='666', type=int, help=' random seed for dataset generation.')
    parser.add_argument('--lamda1', default='1', type=float, help='trade-off between loss')
    parser.add_argument('--lamda2', default='1', type=float, help='trade-off between loss')
    parser.add_argument('--alpha', default='1', type=float, help='alpha value for beta distribution')
    parser.add_argument('--dummynumber', default=0, type=int, help='number of dummy label.')
    parser.add_argument('--shmode', action='store_true')


    parser.add_argument('--cog_name', default='结果偏差', type=str, 
                        choices=['确认偏差', '可用性启发式', '刻板印象', '光环效应', '权威偏见', 
                                '框架效应', '从众效应', '群体内偏爱', '对比效应', '过度自信效应', 
                                '损失厌恶', '结果偏差', '后见之明偏差'],
                        help='cognitive bias name to train')

    # dataset args
    parser.add_argument('--dataset_name', default="cognitive", type=str,
                        choices=["Constraint", "POLITIFACT", "LIAR-PLUS", "POLITIFACT", "cognitive", "cognitive_else_cog_as_nega_houjian"])
    parser.add_argument('--data_path', type=str, default='/path/code/TELLER_label/data') # for 2分类测试集
    # parser.add_argument('--data_path', type=str, default='/home/s2024244143/code_yq/BiasMind/data') # for 原来所有测试集合
    parser.add_argument('--mode', type=str, default='binary', choices=['binary', 'multiple'])
    # choose fewer smale for testing
    parser.add_argument('--num_eval_samples', default=5, type=int)
    parser.add_argument('--shot_number', default=0, type=int)
    parser.add_argument('--save_path', default="/path/code/TELLER_label/data/cognitive/report_all11.json", type=str)
    parser.add_argument('--save_all_path', default='/path/code/TELLER_label/data/cognitive/save', type=str)

    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo",
                        choices=["flan-t5-xxl", "flan-t5-xl", "flan-t5-large", "flan-t5-base", "flan-t5-small",
                                 # flan有5种，lm2, gpt1
                                 "Llama-2-7b-chat-hf",
                                 "Llama-2-13b-chat-hf", "gpt-3.5-turbo"])
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--evi_flag', action="store_true")
    parser.add_argument('--eval_mode', type=str, default='logics', choices=['logics', 'sampling'])

    # the parameters of the logic model
    parser.add_argument('--num_conjuncts', default=50, type=int)
    parser.add_argument('--n_out', default=15, type=int, choices=[2, 15])  # 输出类别数
    parser.add_argument('--delta', default=0.01, type=float)
    parser.add_argument('--weight_init_type', default="normal", type=str, choices=["normal", "uniform"])
    parser.add_argument('--mask_flag', default=-2, type=int, choices=[-2, 0])
    parser.add_argument('--initial_delta', '-initial_delta', type=float, default=0.01,
                        help='initial delta.')

    parser.add_argument('--delta_decay_delay', '-delta_decay_delay', type=int, default=1,
                        help='delta_decay_delay.')

    parser.add_argument('--delta_decay_steps', '-delta_decay_steps', type=int, default=1,
                        help='delta_decay_steps.')
    # 0.01 1.3 -> 25 0.1 1.1
    parser.add_argument('--delta_decay_rate', '-delta_decay_rate', type=float, default=1.1,
                        help='delta_decay_rate.')
    # the logic model type
    parser.add_argument('--type_of_logic_model', default="gnn_logic", type=str, choices=[ "gnn_logic_eo", "hgt_logic", "logic", "gnn_logic","mlp", "tree", "bayes"])

    # the parameters of training the logic model， optimizer, schedule
    parser.add_argument('--SGD', '-sgd', action='store_true', help='use optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--n_steps_per_epoch', default=1, type=int)
    parser.add_argument('--scheduler', '-sch', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'CosLR'])
    parser.add_argument('--step_size', '-stp', type=int, default=20, help='fixed step size for StepLR')

    parser.add_argument('--n_epoch', type=int, default=15, help='the number of epochs')  # default 30
    parser.add_argument('--n_batch_step', type=int, default=10,
                        help='the number of batches per step for delta scheduler')  # default 50

    parser.add_argument('--batchsize', default=16, type=int)


    # parser.add_argument('--gqfile', default="/home/s2024244143/code_yq/TELLER_label/data/houjian_Qwen_merged_cognitive_scores.json",
    #                     type=str) # for 2分类


    # for all

    parser.add_argument('--gqfile', default="/path/code/BiasMind/data/cognitive/Qwen2.5-7B-Instruct_False_simplified_kno5_15_test_negative0.1_merges0805.json", type=str)


    parser.add_argument('--evo_flag', action="store_true") #
    parser.add_argument('--evo_file', default=None, type=str)

    parser.add_argument('--graph_flag', action="store_true") #
    parser.add_argument('--graph_merge', default="tanh_adjustment", type=str, choices=["tanh_adjustment","u46", "absolute_s", "ref_dif", "ref_sem", "multiply"])

    parser.add_argument('--sample_u_flag', action="store_true")

    # loss type
    parser.add_argument('--type_of_loss', default="focal", type=str, choices=["focal", "ce"])
    parser.add_argument('--focal_alpha', default="alpha2", type=str, choices=["alpha1", "alpha2", "alpha3", "alpha4"])
    parser.add_argument('--focal_gamma', default=2.0, type=float, choices=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])


    # save the model
    parser.add_argument('--best_target_ckpoint', default="xx.pt", type=str)
    parser.add_argument('--save_flag', action="store_true")

    # the parameters of decision tree
    parser.add_argument('--max_depth', default=6, type=int, help='max_depth of decision tree')
    parser.add_argument('--max_leaf_nodes', default=30, type=int, help='max_leaf_nodes of decision tree')
    parser.add_argument('--min_weight_fraction_leaf', default=0.01, type=float,
                        help='min_weight_fraction_leaf of decision tree')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    print("\n========in main======")
    print("\n========all ,args======",args)
    #
    print("==========graph_flag=======",args.graph_flag)
    print("==========graph_merge=======", args.graph_merge)

    print("==========type_of_loss=======",args.type_of_loss)

    if args.evi_flag:
        gq_files = ["flan-t5-large_True.json"]
    else:
        print("\n========load gq_files======")


        gq_files = ["/path/code/BiasMind/data/cognitive/Qwen2.5-7B-Instruct_False_simplified_kno5_15_test_negative0.1_merges0805.json"] #new  llm score


    args.save_path = os.path.join(args.data_path, args.dataset_name, args.save_path)

    print("\n========args.save_path======",args.save_path)

    conjuncts = [40]
    args.save_flag = True  # save

    if args.n_out == 2:
        args.mode = 'binary'
    else:
        args.mode = 'multiple'
    # lrs =[1e-3]
    lrs = [1e-3]

    wds = [1e-4]

    seeds = [666]
    initial_deltas = [0.1,1]
    # initial_deltas = [1, 0.1, 0.001, 0.0001]
    final_results_wd_con = {}
    final_results = {}
    for initial_delta in initial_deltas:
        for seed in seeds:
            for lr in lrs:
                for wd in wds:
                    for conjunct in conjuncts:
                        args.initial_delta = initial_delta
                        args.num_conjuncts = conjunct
                        args.weight_decay = wd
                        args.lr = lr
                        args.seed = seed
                        print("======initial_delta:",args.initial_delta)
                        print("======seed:", args.seed)
                        print("======lr:", args.lr)
                        print("======wd:", args.weight_decay)
                        print("======conjunct:",  args.num_conjuncts)
                        exp_name_wd_con = '_'.join(
                            [args.dataset_name, str(args.n_out), str(args.num_conjuncts),str(args.lr), str(args.weight_decay)])
                        final_results_wd_con[exp_name_wd_con] = {}
                        final_results_wd_con[exp_name_wd_con]["reported_metrics"] = {}
                        avg_acc = []
                        for gq_file in gq_files:
                            print("======gq_file======:", gq_file)
                            args.gqfile = gq_file
                            exp_name = '_'.join(
                                [args.dataset_name, str(args.n_out), str(args.num_conjuncts), str(args.weight_decay), args.gqfile])
                            # experiment.set_name(exp_name)
                            experiment = None
                            if args.evo_flag:
                                configure = [('P1', 1), ('P2', 1), ('P3', 1), ('P4', 1), ('P5', 1), ('P6', 1),
                                             ('P7', 3), ('P8', 1), ('P9', 1), ('P11', 1), ('P12', 1), ('P13', 1), ('P15', 1)]

                                if args.evo_file is None:
                                    args.evo_file = re.sub(".json", "_evo.json", gq_file)
                            else:
                                configure = []
                                start = 1
                                end = 106



                                for i in range(start, end + 1):
                                    configure.append((f'P{i}', 1))
                                print("=====configure.len====", len(configure))

                            e = Expert(dataset_name=args.dataset_name, mode=args.mode, data_path=args.data_path,
                                       gq_file=args.gqfile, sq_file=None, model_name=args.model_name,
                                       args=args)

                            reported_test_metrics = e.train_logic(args.num_conjuncts, args.n_out, configure=configure,
                                                                  weight_init_type=args.weight_init_type, args=args, exp=experiment)
                            final_results[exp_name] = reported_test_metrics
                            final_results_wd_con[exp_name_wd_con]["reported_metrics"][exp_name] = reported_test_metrics
                            avg_acc.append(reported_test_metrics["final_acc"])

                        final_results_wd_con[exp_name_wd_con]['avg_acc'] = sum(avg_acc) / len(avg_acc)
    max_para = None
    max_acc = 0
    for key in final_results_wd_con.keys():
        if max_acc < final_results_wd_con[key]['avg_acc']:
            max_para = key
            max_acc = final_results_wd_con[key]['avg_acc']
    print(max_para)
    print(max_acc)

    print("#################################")
    print(final_results_wd_con[max_para]["reported_metrics"])

    write_json_file([final_results_wd_con, final_results], args.save_path)

