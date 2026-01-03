import os.path
from enum import Enum
from typing import List, Dict, Tuple, Any
import torch.optim as optim
import torch
from torch import nn, Tensor
from tqdm import tqdm, trange
from utils.evaluation import acc_compute, calculate_macro_f1
import datetime
import random

Label_Mapping_Rule = {"binary": {'后见之明偏差': 0, '结果偏差': 1}, #, '群体归因错误': 2
                      # "multiple": {'true': 0, 'mostly true': 1, 'half true': 2, 'barely true': 3,
                      #              'false': 4, 'pants fire': 5},
                      "multiple": {'后见之明偏差': 0, '结果偏差': 1} #, '群体归因错误': 2
                      }


def scale(p: float, mask_flag=-2): # 要目的是将 (0, 1) 之间的浮点数映射到 (-1, 1) 之间
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
'''
def transform_org_to_logic(configure, set, gq, mask_flag=-2):
    gq_keys = gq.keys()
    # print("============gq_keys:", gq_keys)
    logics_input = []
    label_input = []
    # pre-define a flag
    # print("==========set：", set)

    # print("configure:", configure)
    for sample in set:
        # print("sample[\"ID\"]：", sample["ID"])
        if str(sample["ID"]) in gq_keys:

            tmp = gq[str(sample["ID"])]

            tmp_keys = tmp.keys()
            output = []

            for p, p_num in configure:
                # print("p, p_num:", p, p_num)。（p1,1 ）
                if p in tmp_keys:
                    if len(tmp[p]) > p_num:
                        random_selection = random.sample(tmp[p], p_num)
                        # print("random_selection:", random_selection)
                        for atom in random_selection:
                            output.append(scale(atom[-1], mask_flag=-2))
                    else:
                        for atom in tmp[p]:
                            output.append(scale(atom[-1], mask_flag=-2)) # atom[-1]是不同谓词实例化后的概率分数
                            # scale 将 (0, 1) 之间的浮点数映射到 (-1, 1) 之间;若 atom[-1] 为空（None），则用 mask_flag=-2 填充。
                        output = output + [mask_flag] * (p_num - len(tmp[p])) # 若实际实例化小于p_num 3，不足的也用-2
                else:
                    output = output + [mask_flag] * p_num

            logics_input.append(output)
            label_input.append(sample['label'])
    return logics_input, label_input
# logics_input 是一个二维列表，每行代表一个样本的逻辑输入。
# label_input 记录每个样本的类别标签。


def batch_generation(logics_input, label_input, mode, batchsize):
    assert len(logics_input) == len(label_input), "produce error when generate data splits"
    # split based on the batchsize
    # print("mode:", mode)
    # print("Label_Mapping_Rule[mode]:", Label_Mapping_Rule[mode])
    label_input = [transform_symbols_to_long(label_input[i:i + batchsize], label_mapping=Label_Mapping_Rule[mode]) for i
                   in range(0, len(label_input), batchsize)]
    logics_input = [torch.tensor(logics_input[i:i + batchsize]) for i in range(0, len(logics_input), batchsize)]

    return [(logics_input[i], label_input[i]) for i in range(len(logics_input))]


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
        # sum: N x Q
        return sum

class Disjunction_Dummy(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            layer_type: SemiSymbolicLayerType,
            delta: float,
            weight_init_type: str = "normal"

    ) -> None:
        # configure: {]
        super(Disjunction_Dummy, self).__init__()

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
        abs_weight = torch.abs(self.weights.T.expand(input.size(0), -1, -1) * input.unsqueeze(-1))
        # max_abs_w: N x Q
        max_abs_w = torch.max(abs_weight, dim=1)[0]
        # max_abs_w = 0.0001
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
        # sum: N x Q
        # print("=====sum.shape======", sum.shape)  # torch.Size([8, 3])
        return sum



class DNF(nn.Module):
    def __init__(
            self,
            num_conjuncts: int,
            n_out: int,
            delta: float,
            configure: list[(str, int)],
            weight_init_type: str = "normal",
            shuffle: bool = True,
            dummy_num: int = 0
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

        if dummy_num:
            c = dummy_num
            self.disjunctions = Disjunction_Dummy(
                in_features=num_conjuncts,  # Q
                out_features=n_out + c,  # R+C  输出类别个数
                layer_type=SemiSymbolicLayerType.DISJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )  # weight R x Q

        else: # 即dummy_num=0 不加虚拟类别
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
        # self.args = args
        # print("============args.type_of_logic_model:", args.type_of_logic_model)
        # print("============args.dummynumber:", args.dummynumber)

        if args.type_of_logic_model == "logic":
            self.logic_model = DNF(num_conjuncts=num_conjuncts, n_out=n_out, delta=delta, configure=configure,  # args.initial_delta= 0.01
                                   weight_init_type=weight_init_type, dummy_num=args.dummynumber).to(device)
        elif args.type_of_logic_model == "mlp":
            self.logic_model = MLP(configure, hidden_size=num_conjuncts, output_size=n_out)
        else:
            print("Wrong name of a logic model")
            exit()

        self.criterion = nn.CrossEntropyLoss()
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
        self.best_test_metrics = {}

    def train(self, train_set, validloader, testloader):  # 主要模型函数！！！增加dummyclass
        self.best_metric = 0  # best accuracy on the validation dataset

        self.best_val_epoch = 1
        current_time = datetime.datetime.now()
        # name rule
        self.args.best_target_ckpoint = "bestmodel"
        dir_save = str(self.args.lr) + str(self.args.weight_decay) + str(self.args.num_conjuncts)
        save_path = os.path.join(self.args.data_path, self.args.dataset_name, dir_save)
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
        delta_scheduler = DeltaDelayedExponentialDecayScheduler(initial_delta=self.args.initial_delta, # 0.01
                                                                delta_decay_delay=self.args.delta_decay_delay,
                                                                delta_decay_steps=self.args.delta_decay_steps,
                                                                delta_decay_rate=self.args.delta_decay_rate)
        for epoch in range(self.args.n_epoch):
            ind_list = [i for i in range(len(train_set[0]))]
            random.shuffle(ind_list)
            train_logics_inputs = [train_set[0][i] for i in ind_list]
            train_label_inputs = [train_set[1][i] for i in ind_list]

            trainloader = batch_generation(train_logics_inputs, train_label_inputs, self.args.mode, self.args.batchsize)

            self.n_batch_step = int(len(trainloader) // 3)

            # start from epoch 1
            epoch = epoch + 1
            self.logic_model.train()
            pt = []
            gt = []
            train_loss = 0
            for batch in tqdm(trainloader, desc='Epoch[{}/{}]'.format(epoch, self.args.n_epoch)):
                inputs, targets = batch[0], batch[1]
                gt.append(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(inputs) # self.logic_model DNF!!
                pt.append(self.obtain_label(outputs.cpu()))

                # new add  目前是只能处理一个新类别，dummytargets = 2
                if self.args.dummynumber:

                    dummyoutputs = outputs
                    for i in range(len(outputs)):
                        nowlabel = targets[i]  # =右边为数据的真实label
                        dummyoutputs[i][nowlabel] = -1e9
                        # 将真实label对应的概率值 设为极小值？防止它在 softmax 之后被选中。这是让第二大概率和新类别一致。我以为直接+一个loss那种，居然是softmax之前就设为极小了
                    dummytargets = torch.ones_like(
                        targets) * self.args.known_class  # 此处args.known_class=6 索引0-5, size[bs, 1] 列向量 ，值均为6 用来表示新类别的索引，

                    outputs = outputs / 1024.0  # 温度缩放
                    dummyoutputs =  dummyoutputs / 1024.0
                    
                    loss1 = self.criterion(outputs, targets)  # 数据K+C 和 真实label 的映射
                    loss2 = self.criterion(dummyoutputs, dummytargets)  # 数据输出概率剔除掉 真实label对应的概率值  和 新类别 的映射
                    loss = self.args.lamda1 * loss1 + self.args.lamda2 * loss2  # 从0.01*loss1来看，数据合成起的作用不会很大，后面2个超参数都是1
                else:
                    loss = self.criterion(outputs, targets) # old

                # new add

                # pt.append(self.obtain_label(outputs.cpu()))
                # loss function need adjustment
                # map the outputs to [0 ,1]
                # outputs  = (outputs+1)/2
                # for multiple classification task
                bb_true = outputs[torch.arange(outputs.size(0)), targets]
                bb = torch.stack([bb_true, -bb_true], dim=1)
                fake_label = torch.zeros(outputs.size(0), dtype=torch.long).to(self.device)
                # loss = self.criterion(outputs, targets) + self.criterion(bb, fake_label)



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
                    # 下面这行是否需要注释掉？
                    delta_scheduler.step(self.logic_model, step=self.step)
            train_loss = train_loss / len(list(trainloader))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()

            # print("gt.shape====",len(gt))
            # print("pt.shape====",len(pt))
            # gt.shape==== 2488 差2倍是因为？
            # pt.shape==== 4976

            train_acc = acc_compute(pt, gt) #
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
            print("Train: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision{:.5f}     Recall:{:.5f}".format(
                train_loss, train_acc,
                train_f1,
                train_p,
                train_r))


            # print metrics for trainset
            val_acc, val_f1 = self.validate(epoch, self.logic_model, validloader)

            test_acc, test_macro_f1, test_macro_precision, test_macro_recall= self.test(
                epoch, self.logic_model,
                testloader)

            # test_acc, test_macro_f1, test_macro_precision, test_macro_recall, test_acc_expdum, test_macro_f1_expdum, test_macro_precision_expdum, test_macro_recall_expdum = self.test_new(
            #     epoch, self.logic_model,
            #     testloader)
            # print(self.logic_model.c)
            if val_acc >= self.best_metric:
                self.best_metric = val_acc
                self.best_epoch = epoch
                state = {
                    'net': self.logic_model.state_dict(),
                    'epoch': epoch,
                    'delta': self.logic_model.conjunctions.delta,
                }
                self.best_test_metrics = {"final_acc_expdum": test_acc_expdum,
                                          "final_f1_expdum": test_macro_f1_expdum,
                                          "final_precision_expdum": test_macro_precision_expdum,
                                          "test_macro_recall_expdum": test_macro_recall_expdum,
                                          "final_acc": test_acc,
                                          "final_f1": test_macro_f1,
                                          "final_precision": test_macro_precision,
                                          "test_macro_recall": test_macro_recall
                                          }

                if self.args.save_flag:
                    torch.save(state, os.path.join(save_path, self.args.best_target_ckpoint + ".pth"))
            if test_acc > self.best_test_metric:
                self.best_test_metric_expdum = test_acc_expdum
                self.best_test_f1_expdum = test_macro_f1_expdum
                self.bset_test_precision_expdum = test_macro_precision_expdum
                self.best_test_recall_expdum = test_macro_recall_expdum

                self.best_test_metric = test_acc
                self.best_test_f1 = test_macro_f1
                self.bset_test_precision = test_macro_precision
                self.best_test_recall = test_macro_recall
            if scheduler is not None and self.args.scheduler != 'CosLR':
                scheduler.step()
        print("Best Val Epoch: {}, Best Val Acc： {:.5f}".format(self.best_epoch, self.best_metric))
        print("Best Test Acc: {:.5f}".format(self.best_test_metric))
        print("-----------------------------Final Testing Results------------------------------------------------")
        print(self.best_test_metrics)
        if self.experiment is not None:
            self.experiment.log_metrics(self.best_test_metrics)
        return self.best_test_metrics

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

    def calidummy(epoch, net, testloader):  #校准时候 测试集的已知类和未知类还要分开
        net.eval()
        CONF_AUC = False  # 不使用 AUC（Area Under Curve） 作为评估指标。
        CONF_DeltaP = True  # 使用 ΔP（Delta Probability） 评估，可能是衡量 已知类别和未知类别的置信度差异
        auclist1 = []
        auclist2 = []
        linspace = [0]

        # closerloader =
        # openloader =

        closelogits = torch.zeros((len(closeset), args.known_class + 1)).cuda()  # 初始化为全 0，稍后填充模型的预测值
        openlogits = torch.zeros((len(openset), args.known_class + 1)).cuda()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(closerloader):  # 测试集的已知类别数据
                inputs, targets = inputs.to(device), targets.to(device)
                batchnum = len(targets)
                logits = net(inputs)
                dummylogit = dummypredict(net, inputs)
                maxdummylogit, _ = torch.max(dummylogit, 1)
                maxdummylogit = maxdummylogit.view(-1, 1)  # C个新类别 选一个概率最大的
                totallogits = torch.cat((logits, maxdummylogit), dim=1)  # 最终预测的 所有类别 概率
                closelogits[batch_idx * batchnum:batch_idx * batchnum + batchnum,
                :] = totallogits  # totallogits 赋值给 closelogits 的对应 batch 位置，用于存储批次预测结果。

            for batch_idx, (inputs, targets) in enumerate(openloader):  # 测试集中未知类别数据，为啥分开？里面代码一样，分开为后面的校准
                inputs, targets = inputs.to(device), targets.to(device)
                batchnum = len(targets)
                logits = net(inputs)
                dummylogit = dummypredict(net, inputs)
                maxdummylogit, _ = torch.max(dummylogit, 1)
                maxdummylogit = maxdummylogit.view(-1, 1)
                totallogits = torch.cat((logits, maxdummylogit), dim=1)
                openlogits[batch_idx * batchnum:batch_idx * batchnum + batchnum, :] = totallogits

        Logitsbatchsize = 200  # 校准的bs是200
        maxauc = 0
        maxaucbias = 0
        for biasitem in linspace:  # linspace=[0] 这是在校准吧，两种方式
            if CONF_AUC:  # 包含温度和偏执缩放两种
                for temperature in [
                    1024.0]:  # 温度缩放 模型校准的一种方法，遍历不同的温度缩放参数；softmax(logits / temperature) 可以平滑化 logits，从而调整置信度。
                    closeconf = []  # 存储校准后的 closed-set数据 dummy 类 置信度的列表
                    openconf = []
                    closeiter = int(len(closelogits) / Logitsbatchsize)  # 计算需要多少个 batch 处理 closelogits
                    openiter = int(len(openlogits) / Logitsbatchsize)
                    for batch_idx in range(closeiter):
                        logitbatch = closelogits[
                                     batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize, :]
                        logitbatch[:, -1] = logitbatch[:, -1] + biasitem  # 校准 预测的dummy 类（未知类别）的 logit。
                        embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                        # 因为这里温度只有1024，会使得 logits 变得非常平滑， 为什么要平滑这么多
                        # 在此处，普通softmax 中指数函数会使得 logits 中较大的值会被放大，导致概率分布变得极端（接近 0 或 1），即已知类别 logits 可能很大，，而高温度会缩放它们，使 softmax 计算出的概率更接近线性。
                        conf = embeddings[:, -1]  # 提取dummy 类 置信度
                        closeconf.append(conf.cpu().numpy())
                    closeconf = np.reshape(np.array(closeconf),
                                           (-1))  # 转换为一维数组 eg. closeconf = [0.1, 0.3, 0.5]  每个元素表示样本 dummy 类 的置信度：
                    closelabel = np.ones_like(closeconf)  # 已知类别的标签设为 1   [1, 1, 1,], dummy 类是0

                    for batch_idx in range(openiter):
                        logitbatch = openlogits[
                                     batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize, :]
                        logitbatch[:, -1] = logitbatch[:, -1] + biasitem
                        embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                        conf = embeddings[:, -1]
                        openconf.append(conf.cpu().numpy())
                    openconf = np.reshape(np.array(openconf), (-1))
                    openlabel = np.zeros_like(openconf)
                    totalbinary = np.hstack([closelabel, openlabel])  # totalbinary = [1, 1, 1, 0, 0, 0]
                    totalconf = np.hstack(
                        [closeconf, openconf])  # totalconf = [ 0.1, 0.3, 0.5，0.8, 0.6, 0.9]：表示每个样本被预测为未知类别的置信度
                    auc1 = roc_auc_score(1 - totalbinary, totalconf)
                    auc2 = roc_auc_score(totalbinary,
                                         totalconf)  # # roc_auc_score(y_true, y_score) 计算 AUC，y_true 是真实标签，y_score 是预测置信度。

                    # auc1 反映模型能否正确地识别未知类别 衡量未知类别的区分能力：
                    # auc2 衡量已知类别的区分能力：

                    print('Temperature:', temperature, 'bias', biasitem, 'AUC_by_confidence', auc2)
                    auclist1.append(np.max([auc1, auc2]))
            if CONF_DeltaP:  # dfault 用的这个 now
                for temperature in [1024.0]:
                    closeconf = []
                    openconf = []
                    closeiter = int(len(closelogits) / Logitsbatchsize)
                    openiter = int(len(openlogits) / Logitsbatchsize)
                    for batch_idx in range(closeiter):
                        logitbatch = closelogits[
                                     batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize, :]
                        logitbatch[:, -1] = logitbatch[:, -1] + biasitem
                        embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                        dummyconf = embeddings[:, -1].view(-1, 1)
                        maxknownconf, _ = torch.max(embeddings[:, :-1], dim=1)  # 计算已知类别（前 K-1 列）的最大置信度。
                        maxknownconf = maxknownconf.view(-1, 1)
                        conf = dummyconf - maxknownconf  # 计算未知类别置信度与已知类别最大置信度之差,如果正+，更像dummy类，用于衡量样本更像已知还是未知类别。
                        closeconf.append(conf.cpu().numpy())
                    closeconf = np.reshape(np.array(closeconf), (-1))
                    closelabel = np.ones_like(closeconf)
                    for batch_idx in range(openiter):
                        logitbatch = openlogits[
                                     batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize, :]
                        logitbatch[:, -1] = logitbatch[:, -1] + biasitem
                        embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                        dummyconf = embeddings[:, -1].view(-1, 1)
                        maxknownconf, _ = torch.max(embeddings[:, :-1], dim=1)
                        maxknownconf = maxknownconf.view(-1, 1)
                        conf = dummyconf - maxknownconf
                        openconf.append(conf.cpu().numpy())
                    openconf = np.reshape(np.array(openconf), (-1))
                    openlabel = np.zeros_like(openconf)
                    totalbinary = np.hstack([closelabel, openlabel])
                    totalconf = np.hstack([closeconf, openconf])  # 可能有负值
                    auc1 = roc_auc_score(1 - totalbinary,
                                         totalconf)  # 现在已知类别 0，未知类别 1。[0,0,0, ..., 1,1,1] 符合165行，如果正，更像dummy类
                    auc2 = roc_auc_score(totalbinary, totalconf)
                    print('Temperature:', temperature, 'bias', biasitem, 'AUC_by_Delta_confidence', auc1)  # 这里用的1
                    auclist1.append(np.max([auc1, auc2]))
        return np.max(np.array(auclist1))

    def validate(self, epoch, net, validloader):
        net.eval()
        pt = []
        gt = []
        loss = 0.0
        with torch.no_grad():
            for batch in validloader:
                inputs, targets = batch[0], batch[1]
                gt.append(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(inputs)
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
        print("Val: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision{:.5f}     Recall:{:.5f}".format(loss, acc,
                                                                                                              macro_f1,
                                                                                                              macro_precision,
                                                                                                              macro_recall))
        return acc, macro_f1


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

    def test_new(self, epoch, net, testloader):  # new
        net.eval()
        pt = []
        gt = []
        loss = 0.0
        with torch.no_grad():
            for batch in testloader:
                inputs, targets = batch[0], batch[1]
                gt.append(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(inputs)
                loss = self.criterion(outputs, targets).item() + loss
                # inter outputs from outputs of self.logic_model
                pt.append(self.obtain_label(outputs.cpu()))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()

        loss = loss / len(testloader)
        # 所有类别的 # Calculate overall accuracy and macro F1 for all classes
        acc = acc_compute(pt, gt)
        macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)

        # 筛选出类别0和类别1的样本
        filtered_gt = [g for g in gt if g != 2]  # 排除类别2
        filtered_pt = [p for p, g in zip(pt, gt) if g != 2]  # 排除类别2的预测

        # 计算类0和类1合并后的准确率、F1、精度、召回率
        acc_expdum = acc_compute(filtered_pt, filtered_gt)
        f1_expdum , precision_expdum , recall_expdum = calculate_macro_f1(filtered_pt, filtered_gt)

        if self.experiment is not None:
            self.experiment.log_metric('{}/test'.format("loss"),
                                       loss, epoch)
            self.experiment.log_metric('{}/test'.format("acc"),
                                       acc, epoch)
            self.experiment.log_metric('{}/test'.format("acc"),
                                       acc_expdum, epoch)
            self.experiment.log_metric('{}/test'.format("macro_F1"),
                                       macro_f1, epoch)
            self.experiment.log_metric('{}/test'.format("acc"),
                                       f1_expdum, epoch)

        print(
            "Test: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision:{:.5f}     Recall:{:.5f} ".format(loss, acc,
                                                                                                              macro_f1,
                                                                                                              macro_precision,
                                                                                                              macro_recall))
        print(
            "Test: Loss {:.5f}       Acc_expdum:{:.5f}       F1_expdum:{:.5f}    Precision_expdum:{:.5f}     Recall_expdum:{:.5f} ".format(loss, acc_expdum,
                                                                                                              f1_expdum,
                                                                                                              precision_expdum,
                                                                                                              recall_expdum))

        return acc, macro_f1, macro_precision, macro_recall, acc_expdum,f1_expdum, precision_expdum, recall_expdum





    def test(self, epoch, net, testloader):  # old
        net.eval()
        pt = []
        gt = []
        loss = 0.0
        with torch.no_grad():
            for batch in testloader:
                inputs, targets = batch[0], batch[1]
                gt.append(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(inputs)
                loss = self.criterion(outputs, targets).item() + loss
                # inter outputs from outputs of self.logic_model
                pt.append(self.obtain_label(outputs.cpu()))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
        loss = loss / len(testloader)
        acc = acc_compute(pt, gt)   # 这里是总的acc 3类的，我希望也能计算2类的，so 后面再议
        macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
        if self.experiment is not None:
            self.experiment.log_metric('{}/test'.format("loss"),
                                       loss, epoch)
            self.experiment.log_metric('{}/test'.format("acc"),
                                       acc, epoch)
            self.experiment.log_metric('{}/test'.format("macro_F1"),
                                       macro_f1, epoch)
        print(
            "Test: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision:{:.5f}     Recall:{:.5f}".format(loss, acc,
                                                                                                              macro_f1,
                                                                                                              macro_precision,
                                                                                                              macro_recall))

        return acc, macro_f1, macro_precision, macro_recall

    def obtain_label(self, logicts: torch.tensor):
        labels = torch.argmax(logicts, dim=1)
        return labels
