import os
import re

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import logging
import numpy as np
from utils.data_reading import load_data_for_expert
from models.qa_t5 import T5_Question_Answering, FT5_VARIANT, GPT_VARIANT, LLAMA2_VARIANT
import argparse
from utils.evaluation import acc_compute, calculate_macro_f1
from sklearn.metrics import recall_score
import torch
from utils.components.dnf_layer import batch_generation, transform_org_to_logic, DNF, GNN_DNF, nn, SemiSymbolicLayerType
from focal_loss import MultiClassFocalLoss
from tqdm import tqdm

# python drive_prune.py

log = logging.getLogger()
logging.basicConfig(level="INFO")

alpha_prune = [1.1135856097836927, 1.1655539811342945, 1.2189099842720732, 1.2498730000264116, 1.2533898762916564,
               1.2797705794659595, 1.3006006695679493, 1.31009460877715, 1.4326544099239886, 1.4388577414488666,
               1.5550319074664687, 1.5713080210486097, 1.6002287078920014, 1.6323716204658443, 1.6834580875735297]



def prune_layer_weight_topk(
        model: DNF,
        layer_type: SemiSymbolicLayerType,
        epsilon: float,
        device,
        data_loader,
        show_tqdm: bool = True,
) -> int:
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = model.conjunctions.weights.data.T.clone()
    else:
        curr_weight = model.disjunctions.weights.data.clone()

    og_perf = test_dnf(model, data_loader, device)
    prune_count = 0
    weight_device = curr_weight.device

    flatten_weight_len = curr_weight.numel()
    log.info("The number of weights that need to be tested one by one in the current {}".format(layer_type, flatten_weight_len))
    iterator = tqdm(range(flatten_weight_len)) if show_tqdm else range(flatten_weight_len)


    protected_idx_set = set()
    if layer_type == SemiSymbolicLayerType.DISJUNCTION:
        num_classes, num_features = curr_weight.shape
        perf_impact = torch.full_like(curr_weight, float('-inf'))  # The importance of storing each weight

        for row in tqdm(range(num_classes), desc="Calculate the importance of each class weight") if show_tqdm else range(num_classes):
            for col in range(num_features):
                if curr_weight[row, col] == 0:
                    continue
                temp_weight = curr_weight.clone()
                temp_weight[row, col] = 0

                model.disjunctions.weights.data = temp_weight
                new_perf = test_dnf(model, data_loader, device)
                perf_drop = og_perf - new_perf
                perf_impact[row, col] = perf_drop


        for row in range(num_classes):
            topk = torch.topk(perf_impact[row], k=3)
            for col_idx in topk.indices.tolist():
                protected_idx_set.add(row * num_features + col_idx)
        model.disjunctions.weights.data = curr_weight


    for i in iterator:
        flat_weight = curr_weight.reshape(-1)
        if flat_weight[i] == 0:
            continue

        if (layer_type == SemiSymbolicLayerType.DISJUNCTION) and (i in protected_idx_set):
            continue


        mask = torch.ones_like(flat_weight)
        mask[i] = 0
        masked_weight = (flat_weight * mask).reshape(curr_weight.shape)


        if layer_type == SemiSymbolicLayerType.CONJUNCTION:
            model.conjunctions.weights.data = masked_weight.T
        else:
            model.disjunctions.weights.data = masked_weight


        new_perf = test_dnf(model, data_loader, device)
        performance_drop = og_perf - new_perf

        if performance_drop < epsilon:
            prune_count += 1
            curr_weight.reshape(-1)[i] = 0
        else:

            if layer_type == SemiSymbolicLayerType.CONJUNCTION:
                model.conjunctions.weights.data = curr_weight.T
            else:
                model.disjunctions.weights.data = curr_weight


    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        model.conjunctions.weights.data = curr_weight.T
    else:
        model.disjunctions.weights.data = curr_weight

    log.info("The number of weights subtracted in the current step:{}".format(prune_count))
    log.info("====Model parameters have been updated! Current model performance:")
    test_dnf(model, data_loader, device)


    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        log.info("====================Currently, with special pruning of the conjunctive layer and topk constraint, the model's current performance is as follows:")

        original_perf = test_dnf(model, data_loader, device)

        weight_matrix = model.conjunctions.weights.data.T.clone()  # [100, 106]
        prune_counter = 0
        weight_device = weight_matrix.device

        num_rows, num_cols = weight_matrix.shape  # (100, 106)
        row_iterator = tqdm(range(num_rows), desc="Pruning rows") if show_tqdm else range(num_rows)

        for row_idx in row_iterator:
            row_weights = weight_matrix[row_idx].clone()
            non_zero_mask = row_weights != 0
            original_non_zero = non_zero_mask.sum().item()


            if original_non_zero <= 3:
                print("The con layer already has {} nodes, so there's no need to continue performing topk pruning.".format(original_non_zero))
                continue


            importance_scores = []


            for col_idx in range(num_cols):
                if row_weights[col_idx] == 0:
                    importance_scores.append(-np.inf)
                    continue


                temp_mask = torch.ones_like(row_weights)
                temp_mask[col_idx] = 0
                masked_row = row_weights * temp_mask


                weight_matrix[row_idx] = masked_row
                if layer_type == SemiSymbolicLayerType.CONJUNCTION:
                    model.conjunctions.weights.data = weight_matrix.T
                else:
                    model.disjunctions.weights.data = weight_matrix


                current_perf = test_dnf(model, data_loader, device)
                importance = original_perf - current_perf  # 数值越大表示越重要
                importance_scores.append(importance)

                weight_matrix[row_idx] = row_weights


            topk_indices = np.argsort(importance_scores)[-3:]  # 取影响最大的3个


            final_mask = torch.zeros_like(row_weights)
            final_mask[topk_indices] = 1
            pruned_row = row_weights * final_mask


            current_non_zero = (pruned_row != 0).sum().item()
            prune_counter += (original_non_zero - current_non_zero)


            weight_matrix[row_idx] = pruned_row


        model.conjunctions.weights.data = weight_matrix.T

        log.info("==============The special pruning of the conjunctive layer ends, and the model's performance improves.：")
        final_perf = test_dnf(model, data_loader, device)
        performance_drop = original_perf - final_perf

        if performance_drop > epsilon:
            log.info(f"Performance degradation exceeds the threshold: {performance_drop:.4f} > {epsilon}")


        return prune_counter + prune_count
    else:

        return prune_count






def apply_threshold(
        model: GNN_DNF,
        og_conj_weight,
        og_disj_weight,
        t_val,
        const: float = 6.0,
) -> None:
    new_conj_weight = (
            (torch.abs(og_conj_weight) > t_val) * torch.sign(og_conj_weight) * const
    )
    model.conjunctions.weights.data = new_conj_weight

    new_disj_weight = (
            (torch.abs(og_disj_weight) > t_val) * torch.sign(og_disj_weight) * const
    )
    model.disjunctions.weights.data = new_disj_weight


def batch_iter(configure, s_set, gq, mask_flag, mode, batchsize):
    text_inputs, logics_input, label_input = transform_org_to_logic(configure, s_set, gq,
                                                                    mask_flag=mask_flag)
    # print("=============logics_input, label_input:", logics_input, label_input)
    loader = batch_generation(text_inputs, logics_input, label_input, mode, batchsize)
    return loader


def obtain_label(logicts: torch.tensor):
    labels = torch.argmax(logicts, dim=1)
    return labels


def extract_asp_rules(sd: dict, flatten: bool = False):
    print("################extract_asp_rules#####################")

    output_rules = []

    # Get all conjunctions Q \times P
    # P input_dim, Q the number of conjunctions
    conj_w = sd["conjunctions.weights"].T

    conjunction_map = dict()
    for i, w in enumerate(conj_w):


        if torch.all(w == 0):
            # No conjunction is applied here
            print("没有合取项，No conjunction is applied here for {}!!!".format(i))
            continue

        conjuncts = []
        for j, v in enumerate(w):
            # print("j（谓词P）:{}; v:{}".format(j, v))

            if v < 0:
                # Negative weight, negate the atom
                conjuncts.append(f"not has_attr_{j + 1}")
            elif v > 0:
                # Positive weight, normal atom
                conjuncts.append(f"has_attr_{j + 1}")

        conjunction_map[i] = conjuncts

    # Get DNF
    # Get all conjunctions Y \times Q
    disj_w = sd["disjunctions.weights"]

    not_covered_classes = []
    for i, w in enumerate(disj_w):
        # print("i:{}; w:{}".format(i, w))
        if torch.all(w == 0):
            # No DNF for class i
            not_covered_classes.append(i)
            print("No DNF for class {}!!!".format(i))
            continue

        disjuncts = []
        for j, v in enumerate(w):
            # print("j:{}; v:{}".format(j, v))
            if v < 0 and j in conjunction_map:
                # Negative weight, negate the existing conjunction

                if flatten:

                    # Need to add auxiliary predicate (conj_X) which is not yet
                    # in the final rules list
                    ttt = f"conj_{j} :- {', '.join(conjunction_map[j])}."
                    if ttt not in output_rules:
                        output_rules.append(
                            ttt
                        )
                    output_rules.append(f"label({i}) :- not conj_{j}.")
                else:

                    disjuncts.append(f"not conj_{j}")
            elif v > 0 and j in conjunction_map:
                # Positive weight, add normal conjunction

                if flatten:
                    ttt = f"conj_{j} :- {', '.join(conjunction_map[j])}."
                    if ttt not in output_rules:
                        output_rules.append(
                            ttt
                        )
                    output_rules.append(f"label({i}) :- conj_{j}.")
                else:
                    disjuncts.append(f"conj_{j}")

        if not flatten:
            for disjunct in disjuncts:
                output_rules.append(f"label({i}) :- {disjunct}.")

    return output_rules


def calculate_class_f1(pt, gt, class_idx=1):
    """
    Calculates the F1 score, precision, and recall for the specified class.

    Parameters:

    pt: Predicted labels (numpy array)

    gt: True labels (numpy array)

    class_idx: The index of the class to calculate (0 indicates the first class)

    Returns:

    f1: F1 score for the specified class

    precision: Precision for the specified class

    recall: Recall for the specified class
    """

    pt = np.array(pt)
    gt = np.array(gt)


    tp = np.sum((pt == class_idx) & (gt == class_idx))
    fp = np.sum((pt == class_idx) & (gt != class_idx))
    fn = np.sum((pt != class_idx) & (gt == class_idx))


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0


    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall


def recall_label1(pt: list, gt: list) -> float:
    """

    """
    label_predicted = [p == 1 for p in pt]
    label_true = [t == 1 for t in gt]
    return recall_score(label_true, label_predicted)


def test_dnf(logic_model, testloader, device):
    criterion = MultiClassFocalLoss(alpha=alpha_prune, gamma=2.0, reduction='mean')
    logic_model.eval()
    pt = []
    gt = []
    loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            texts, masks, inputs, targets = batch[0], batch[1], batch[2], batch[3]

            gt.append(targets)
            texts, masks, inputs, targets = texts.to(device), masks.to(device), inputs.to(
                device), targets.to(device)

            # outputs, saved_variable = logic_model(texts, masks, inputs, "test")
            outputs, saved_variable = logic_model(texts, masks, inputs)
            loss = criterion(outputs, targets).item() + loss
            # inter outputs from outputs of self.logic_model
            pt.append(obtain_label(outputs.cpu()))

        gt = torch.cat(gt).tolist()
        pt = torch.cat(pt).tolist()
    acc = acc_compute(pt, gt)
    loss = loss / len(testloader)
    macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)


    print(
        "Val_or_Test:Loss:{:.5f}, Acc:{:.5f}       F1:{:.5f}    Precision:{:.5f}     Recall:{:.5f}\n".format(loss, acc,
                                                                                                             macro_f1,
                                                                                                             macro_precision,
                                                                                                             macro_recall))
    # return acc old
    return acc



# new
def prune_layer_weight_new(
        model: GNN_DNF,
        layer_type: SemiSymbolicLayerType,
        epsilon,
        device,
        data_loader,
        show_tqdm=True,
) -> int:
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = model.conjunctions.weights.data.T.clone()
    else:
        curr_weight = model.disjunctions.weights.data.clone()

    og_perf = test_dnf(model, data_loader, device)
    original_weight = curr_weight.clone()
    prune_count = 0

    if layer_type == SemiSymbolicLayerType.DISJUNCTION:
        n_classes, n_conj = curr_weight.shape

        keep_mask = torch.zeros_like(curr_weight, dtype=bool)
        for i in range(n_classes):
            if torch.all(curr_weight[i] == 0):
                continue
            j_max = torch.argmax(torch.abs(curr_weight[i]))
            keep_mask[i, j_max] = True


        non_keep_mask = ~keep_mask
        non_keep_weights = curr_weight[non_keep_mask]
        if len(non_keep_weights) == 0:
            return 0


        abs_weights = torch.abs(non_keep_weights)
        low, high = 0, torch.max(abs_weights).item()
        best_t = 0
        best_prune = 0
        for _ in range(20):
            mid = (low + high) / 2
            temp_weight = original_weight.clone()
            prune_mask = (torch.abs(temp_weight) <= mid) & non_keep_mask
            temp_weight[prune_mask] = 0
            model.disjunctions.weights.data = temp_weight
            new_perf = test_dnf(model, data_loader, device)
            if og_perf - new_perf <= epsilon:
                best_t = mid
                best_prune = prune_mask.sum().item()
                low = mid
            else:
                high = mid
            if high - low < 1e-6:
                break


        prune_mask = (torch.abs(original_weight) <= best_t) & non_keep_mask
        curr_weight[prune_mask] = 0
        prune_count = prune_mask.sum().item()
        model.disjunctions.weights.data = curr_weight

    else:
        abs_weights = torch.abs(original_weight)
        total_elements = abs_weights.numel()
        max_possible_k = min(5, total_elements)
        best_k = 0
        best_prune_count = 0


        for k in range(max_possible_k, -1, -1):
            if k == 0:
                temp_weight = torch.zeros_like(original_weight)
            else:

                topk_values, topk_indices = torch.topk(
                    abs_weights.flatten(), k, largest=True
                )
                keep_mask = torch.zeros_like(abs_weights.flatten(), dtype=torch.bool)
                keep_mask[topk_indices] = True
                keep_mask = keep_mask.reshape(abs_weights.shape)
                temp_weight = original_weight.clone()
                temp_weight[~keep_mask] = 0


            model.conjunctions.weights.data = temp_weight.T
            new_perf = test_dnf(model, data_loader, device)
            model.conjunctions.weights.data = original_weight.T

            if og_perf - new_perf <= epsilon:
                best_k = k
                best_prune_count = total_elements - k if k > 0 else total_elements
                break


        if best_k > 0:
            topk_values, topk_indices = torch.topk(abs_weights.flatten(), best_k)
            keep_mask = torch.zeros_like(abs_weights.flatten(), dtype=torch.bool)
            keep_mask[topk_indices] = True
            keep_mask = keep_mask.reshape(abs_weights.shape)
            curr_weight[~keep_mask] = 0
            prune_count = best_prune_count
        else:

            prune_count = 0

        model.conjunctions.weights.data = curr_weight.T

    return prune_count


# old
def prune_layer_weight(
        model: GNN_DNF,
        layer_type: SemiSymbolicLayerType,
        epsilon,
        device,
        data_loader,
        show_tqdm=True,
) -> int:
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = model.conjunctions.weights.data.T.clone()
    else:
        curr_weight = model.disjunctions.weights.data.clone()

    og_perf = test_dnf(model, data_loader, device)

    prune_count = 0
    weight_device = curr_weight.device

    flatten_weight_len = len(torch.reshape(curr_weight, (-1,)))

    base_iterator = range(flatten_weight_len)
    iterator = tqdm(base_iterator, desc="iterator in prune_layer_weight") if show_tqdm else base_iterator
    # Traverse each weight
    for i in iterator:
        curr_weight_flatten = torch.reshape(curr_weight, (-1,))

        if curr_weight_flatten[i] == 0:
            continue

        mask = torch.ones(flatten_weight_len, device=weight_device)
        mask[i] = 0
        mask = mask.reshape(curr_weight.shape)

        masked_weight = curr_weight * mask

        if layer_type == SemiSymbolicLayerType.CONJUNCTION:
            model.conjunctions.weights.data = masked_weight.T
        else:
            model.disjunctions.weights.data = masked_weight

        new_perf = test_dnf(model, data_loader, device)
        performance_drop = og_perf - new_perf

        if performance_drop < epsilon:
            prune_count += 1
            curr_weight *= mask

    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        model.conjunctions.weights.data = curr_weight.T
    else:
        model.disjunctions.weights.data = curr_weight
    return prune_count



def prune_layer_weight_dynamic(
        model: GNN_DNF,
        layer_type: SemiSymbolicLayerType,
        epsilon,
        device,
        data_loader,
        show_tqdm=True,
) -> int:
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = model.conjunctions.weights.data.T.clone()
    else:
        curr_weight = model.disjunctions.weights.data.clone()

    og_perf = test_dnf(model, data_loader, device)

    prune_count = 0
    weight_device = curr_weight.device

    flatten_weight_len = len(torch.reshape(curr_weight, (-1,)))
    base_iterator = range(flatten_weight_len)
    iterator = tqdm(base_iterator, desc="iterator in prune_layer_weight") if show_tqdm else base_iterator
    # Traverse each weight
    for i in iterator:
        curr_weight_flatten = torch.reshape(curr_weight, (-1,))

        if curr_weight_flatten[i] == 0:
            continue

        mask = torch.ones(flatten_weight_len, device=weight_device)
        mask[i] = 0
        mask = mask.reshape(curr_weight.shape)

        masked_weight = curr_weight * mask

        if layer_type == SemiSymbolicLayerType.CONJUNCTION:
            model.conjunctions.weights.data = masked_weight.T
        else:
            model.disjunctions.weights.data = masked_weight

        new_perf = test_dnf(model, data_loader, device)
        performance_drop = og_perf - new_perf

        if curr_weight_flatten[i] < 0:
            if performance_drop < epsilon * 3:  # epsilon * 2
                prune_count += 1
                curr_weight *= mask
        else:
            if performance_drop < epsilon / 3:  # epsilon / 2
                prune_count += 1
                curr_weight *= mask

    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        model.conjunctions.weights.data = curr_weight.T
    else:
        model.disjunctions.weights.data = curr_weight
    return prune_count


def remove_unused_conjunctions(model: GNN_DNF) -> int:
    disj_w = model.disjunctions.weights.data.clone()
    unused_count = 0

    for i, w in enumerate(disj_w.T):
        if torch.all(w == 0):
            # The conjunction is not used at all
            model.conjunctions.weights.data[:, i] = 0
            unused_count += 1

    return unused_count


def remove_disjunctions_when_empty_conjunctions(model: GNN_DNF) -> int:
    # If a conjunction has all 0 weights (no input atom is used), then this
    # conjunction shouldn't be used in a rule.
    conj_w = model.conjunctions.weights.T.data.clone()
    unused_count = 0

    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # This conjunction should not be used
            model.disjunctions.weights.data[:, i] = 0
            unused_count += model.disjunctions.weights.shape[0]

    return unused_count


class Prune:
    def __init__(self, dataset_name, mode, data_path, gq_file, sq_file, model_name, args):
        # prepare data
        self.dataset_name = dataset_name
        self.mode = mode
        self.evo_flag = args.evo_flag
        self.data_path = os.path.join(data_path, self.dataset_name)
        self.gq_file = gq_file
        self.sq_file = sq_file
        self.evo_file = args.evo_file
        self.model_name = model_name
        self.args = args
        self.dataset, self.rule = load_data_for_expert(data_path=self.data_path, dataset_name=self.dataset_name,
                                                       mode=self.mode, gq_file=self.gq_file, sq_file=self.sq_file,
                                                       evo_file=self.evo_file, evo_flag=self.evo_flag)
        self.save_path = args.save_path

        # lode predicates set
        self.predicate_set = {}
        for a in configure:
            self.predicate_set[a[0]] = a[1]
        #  lode the data
        train_set = self.dataset["train"]
        val_set = self.dataset["val"]
        test_set = self.dataset["test"]
        gq = self.dataset["gq"]
        train_text_inputs, train_logics_inputs, train_label_inputs = transform_org_to_logic(configure, train_set, gq,
                                                                                            mask_flag=args.mask_flag)
        train_set = [train_text_inputs, train_logics_inputs, train_label_inputs]

        ind_list = [i for i in range(len(train_set[0]))]

        train_text_inputs = [train_set[0][i] for i in ind_list]
        train_logics_inputs = [train_set[1][i] for i in ind_list]
        train_label_inputs = [train_set[2][i] for i in ind_list]

        self.val_loader = batch_iter(configure, val_set, gq, mask_flag=args.mask_flag, mode=self.mode,
                                     batchsize=args.batchsize)
        self.test_loader = batch_iter(configure, test_set, gq, mask_flag=args.mask_flag, mode=self.mode,
                                      batchsize=args.batchsize)
        # self.trainloader = batch_generation(train_logics_inputs, train_label_inputs, self.args.mode,
        #                                     self.args.batchsize)

        self.trainloader = batch_generation(train_text_inputs, train_logics_inputs, train_label_inputs, self.args.mode,
                                            self.args.batchsize)

        #   batch_generation返回 [(text_ids[i], attention_masks[i], logics_input[i], label_input[i]) for i in range(len(logics_input))]
        # for pruning
        self.result_dict = dict()
        self.device = self.args.device if torch.cuda.is_available() else 'cpu'

        # Post-training process parameters
        self.prune_epsilon: float = 0.001  # default 0.005;;;  不太对 permitted performance drop after tuning
        self.tune_epochs: int = 10  # default 100
        self.tune_weight_constraint_lambda: float = 0.005
        # load the model
        self.pth_file_base_name = os.path.join(args.data_path, args.dataset_name, args.best_dir,
                                               args.best_target_ckpoint)
        # tune the model
        self.optimiser_fn = lambda params: torch.optim.Adam(
            params, lr=args.lr, weight_decay=self.args.weight_decay
        )
        self.criterion = MultiClassFocalLoss(alpha=alpha_prune, gamma=2.0, reduction='mean')

    def _after_train_eval(self, model: GNN_DNF) -> None:
        log.info("DNF performance after train")
        acc = test_dnf(
            model, self.val_loader, self.device
        )
        log.info(f"\nDNF Testing Acc: {acc:.3f}\n")
        self.result_dict["after_train_test"] = round(acc, 3)

    def _pruning(self, model: GNN_DNF) -> None:
        new_perf_test = test_dnf(model, self.test_loader, self.device)

        log.info(f"Initial prune (test): {new_perf_test:.5f}\n")

        # Pruning procedure:
        # 1. Prune disjunction
        # 2. Prune unused conjunctions
        #   - If a conjunction is not used in any disjunctions, pruned the
        #     entire disjunct body
        # 3. Prune conjunctions
        # 4. Prune disjunctions that uses empty conjunctions
        #   - If a conjunction has no conjunct, no disjunctions should use it
        # 5. Prune disjunction again
        log.info("Pruning on DNF starts")

        # 1. Prune disjunction  先剪xiqu层
        log.info("Prune disj layer")
        # prune_count = prune_layer_weight(
        #     model,
        #     SemiSymbolicLayerType.DISJUNCTION,
        #     self.prune_epsilon,
        #     self.device,
        #     self.val_loader
        # )

        # 只对xiqu层引用dynamic
        prune_count = prune_layer_weight_dynamic(
            model,
            SemiSymbolicLayerType.DISJUNCTION,
            self.prune_epsilon,
            self.device,
            self.val_loader
        )

        new_perf = test_dnf(model, self.val_loader, self.device)

        log.info(f"Pruned disj count (1st):   {prune_count}")
        log.info(f"New perf after disj:       {new_perf:.3f}")

        # 2. Prune unused conjunctions
        unused_conj = remove_unused_conjunctions(model)
        log.info(f"Remove unused conjunctions: {unused_conj}")

        # 3. Prune conjunctions
        log.info("Prune conj layer")
        prune_count = prune_layer_weight(
            model,
            SemiSymbolicLayerType.CONJUNCTION,
            self.prune_epsilon,
            self.device,
            self.val_loader
        )
        new_perf = test_dnf(model, self.val_loader, self.device)
        log.info(f"Pruned conj count:           {prune_count}")
        log.info(f"New perf after conj:         {new_perf:.3f}")

        # 4. Prune disjunctions that uses empty conjunctions
        removed_disj = remove_disjunctions_when_empty_conjunctions(model)
        log.info(
            f"Remove disjunction that uses empty conjunctions: {removed_disj}"
        )

        # 5. Prune disjunction again
        log.info("Prune disj layer again")
        # prune_count = prune_layer_weight(
        #     model,
        #     SemiSymbolicLayerType.DISJUNCTION,
        #     self.prune_epsilon,
        #     self.device,
        #     self.val_loader
        # )

        prune_count = prune_layer_weight_dynamic(
            model,
            SemiSymbolicLayerType.DISJUNCTION,
            self.prune_epsilon,
            self.device,
            self.val_loader
        )

        new_perf = test_dnf(model, self.val_loader, self.device)
        new_perf_test = test_dnf(model, self.test_loader, self.device)
        log.info(f"Pruned disj count (2nd):   {prune_count}")
        log.info(f"New perf after disj (2nd): {new_perf:.3f}")
        log.info(f"New perf after prune (test): {new_perf_test:.3f}\n")

        torch.save(model.state_dict(), self.pth_file_base_name + "_pruned.pth")
        self.result_dict["after_prune_val"] = round(new_perf, 3)
        self.result_dict["after_prune_test"] = round(new_perf_test, 3)

    def _tuning(self, model: GNN_DNF) -> None:
        log.info("Tuning of DNF start")

        initial_cjw = model.conjunctions.weights.data.clone()
        initial_djw = model.disjunctions.weights.data.clone()

        cjw_mask = torch.where(initial_cjw != 0, 1, 0)
        djw_mask = torch.where(initial_djw != 0, 1, 0)

        cjw_inverse_mask = torch.where(initial_cjw != 0, 0, 1)
        djw_inverse_mask = torch.where(initial_djw != 0, 0, 1)

        weight_device = initial_cjw.device

        model.conj_weight_mask = cjw_mask.to(weight_device)
        model.disj_weight_mask = djw_mask.to(weight_device)

        # Weight pushing loss
        def dnf_weight_pushing_constraint():
            # The loss should be only applied to not pruned weights
            conj_non_zero_w = torch.masked_select(
                model.conjunctions.weights.data,
                model.conj_weight_mask.bool(),
            )
            disj_non_zero_w = torch.masked_select(
                model.disjunctions.weights.data,
                model.disj_weight_mask.bool(),
            )

            def _constraint(w):
                # Pushing the weight to 6/-6/0
                # w * |6 - |w||
                return torch.abs(w * (6 - torch.abs(w))).sum()

            return _constraint(conj_non_zero_w) + _constraint(disj_non_zero_w)

        # Other setup
        optimizer = self.optimiser_fn(model.parameters())

        for epoch in range(self.tune_epochs):
            pt = []
            gt = []
            train_loss = 0
            model.train()
            for batch in self.trainloader:
                assert torch.all(
                    torch.masked_select(
                        model.conjunctions.weights.data,
                        cjw_inverse_mask.bool().to(weight_device),
                    )
                    == 0
                )
                assert torch.all(
                    torch.masked_select(
                        model.disjunctions.weights.data,
                        djw_inverse_mask.bool().to(weight_device),
                    )
                    == 0
                )

                optimizer.zero_grad()

                texts, masks, inputs, targets = batch[0], batch[1], batch[2], batch[3]

                gt.append(targets)
                texts, masks, inputs, targets = texts.to(self.device), masks.to(self.device), inputs.to(
                    self.device), targets.to(self.device)  # 这里inputs 是

                outputs, saved_variable = model(texts, masks, inputs)
                pt.append(obtain_label(outputs.cpu()))
                bb_true = outputs[torch.arange(outputs.size(0)), targets]
                bb = torch.stack([bb_true, -bb_true], dim=1)
                fake_label = torch.zeros(outputs.size(0), dtype=torch.long).to(self.device)
                loss = self.criterion(outputs, targets) + self.criterion(bb, fake_label)

                wc = dnf_weight_pushing_constraint()
                loss = (
                               1 - self.tune_weight_constraint_lambda
                       ) * loss + self.tune_weight_constraint_lambda * wc

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # Maintain the pruned weights stay as 0
                model.update_weight_wrt_mask()
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
            train_acc = acc_compute(pt, gt)
            train_loss = train_loss / len(list(self.trainloader))
            log.info(
                "[%3d] Finetune  avg loss: %.3f  avg perf: %.3f"
                % (
                    epoch + 1,
                    train_loss,
                    train_acc,
                )
            )

        perf = test_dnf(model, self.val_loader, self.device)
        log.info(f"Acc after tune: {perf:.3f}")

        torch.save(model.state_dict(), self.pth_file_base_name + "_tuned.pth")

        self.result_dict["after_tune_test"] = round(perf, 3)

    def _thresholding(self, model: GNN_DNF):

        log.info("Thresholding on DNF starts")

        new_perf_test = test_dnf(model, self.test_loader, self.device)

        log.info(f"Initial thero (test): {new_perf_test:.5f}\n")

        conj_min = torch.min(model.conjunctions.weights.data)
        conj_max = torch.max(model.conjunctions.weights.data)
        disj_min = torch.min(model.disjunctions.weights.data)
        disj_max = torch.max(model.disjunctions.weights.data)

        print("============ conj_min, conj_max, disj_min, disj_max:", conj_min, conj_max, disj_min, disj_max)

        threshold_upper_bound = round(
            (
                    # torch.Tensor([conj_min, conj_max, disj_min, disj_max])
                    torch.Tensor([conj_min, conj_max, disj_min, disj_max])
                    .abs()
                    .max()
                    + 0.01
                # - 0.002
            ).item(),
            2,
        )

        print("======Threshold upper bound:", threshold_upper_bound)

        threshold_lower_bound = round(
            (
                    torch.Tensor([conj_min, disj_min])
                    .min()
                    - 0.01
            ).item(),
            2,
        )

        # print("=======Threshold lower bound:", threshold_lower_bound)

        og_conj_weight = model.conjunctions.weights.data.clone()
        og_disj_weight = model.disjunctions.weights.data.clone()


        # old
        perf_scores = []
        t_vals = torch.arange(0, threshold_upper_bound, 0.01)


        # old
        for v in t_vals:
            apply_threshold(model, og_conj_weight, og_disj_weight, v, 6.0)
            perf = test_dnf(model, self.val_loader, self.device)
            perf_scores.append(perf)

        best_jacc_score = max(perf_scores)
        best_t = t_vals[torch.argmax(torch.Tensor(perf_scores))]


        paired = list(zip(perf_scores, t_vals.tolist()))


        top_20 = sorted(paired, key=lambda x: x[0], reverse=True)[:20]


        print("Top 20 performance scores and corresponding t_vals:")
        for i, (score, t_val) in enumerate(top_20):
            print(f"{i + 1:2d}. Score: {score:.5f}, t_val: {t_val:.5f}")

        log.info(
            f"Best t: {best_t.item():.5f}    "
            f"Macro Acc: {best_jacc_score:.5f}"
        )

        apply_threshold(model, og_conj_weight, og_disj_weight, best_t)




        conj_min = torch.min(model.conjunctions.weights.data)
        conj_max = torch.max(model.conjunctions.weights.data)
        disj_min = torch.min(model.disjunctions.weights.data)
        disj_max = torch.max(model.disjunctions.weights.data)

        print("============After  conj_min, conj_max, disj_min, disj_max:", conj_min, conj_max, disj_min, disj_max)

        val_perf = test_dnf(model, self.val_loader, self.device)
        test_perf = test_dnf(model, self.test_loader, self.device)

        log.info(
            f"Val Acc after threshold:  {val_perf:.3f}\n"
        )
        log.info(
            f"Test Acc after threshold: {test_perf:.3f}\n"
        )

        torch.save(
            model.state_dict(), self.pth_file_base_name + "_thresholded.pth"
        )

        self.result_dict["after_threshold_val"] = round(val_perf, 3)
        self.result_dict["after_threshold_test"] = round(test_perf, 3)
        new_perf_test = test_dnf(model, self.test_loader, self.device)

        log.info(f"Final thrro (test): {new_perf_test:.5f}\n")

    def _extract_rules(self, model: GNN_DNF) -> None:

        log.info("Rule extraction starts")
        # print("=========此时的model.state_dict()：", model.state_dict())
        log.info("Rules:")

        rules = extract_asp_rules(model.state_dict(), flatten=True)
        for r in rules:
            log.info(r)

        with open(self.pth_file_base_name + "rules.txt", "w") as f:
            f.write("\n".join(rules))
        print(rules)
        return rules

    def intervent(self, model: GNN_DNF):
        # model.conjunctions.weights.data[5, :] = 0
        model.disjunctions.weights.data[1, 43] = 1
        model.disjunctions.weights.data[1, 34] = 1

    def post_processing(self, model: GNN_DNF):
        log.info("\n------- Post Processing -------")
        prune_num = 2  # 50
        last_rule_num = torch.inf
        self._after_train_eval(model)
        # test_dnf(model, self.test_loader, self.device)
        # for i in tqdm(range(prune_num), desc="range(prune_num) prune_num = 1 "):
        for i in range(prune_num):

            self._pruning(model)
            # self._tuning(model)
            self._thresholding(model)


            rules = self._extract_rules(model)
            now_rule_num = len(rules)
            print("last_rule_num:", last_rule_num)
            print("now_rule_num：{}".format(i, now_rule_num))
            if now_rule_num == last_rule_num:
                print("After {} rounds of pruning, the total number of rules remains unchanged, and the pruning process ends.".format(i))
                break
            if now_rule_num < last_rule_num:
                last_rule_num = now_rule_num
        return self.result_dict


def parse_args():
    parser = argparse.ArgumentParser()

    # proser args new add
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='Proser', type=str, help='Recognition Method')
    # parser.add_argument('--backbone', default='WideResnet', type=str, help='Backbone type.')

    parser.add_argument('--known_class', default=2, type=int, help='number of known class')
    parser.add_argument('--seed', default='42', type=int, help='9 random seed for dataset generation.')
    parser.add_argument('--lamda1', default='1', type=float, help='trade-off between loss')
    parser.add_argument('--lamda2', default='1', type=float, help='trade-off between loss')
    parser.add_argument('--alpha', default='1', type=float, help='alpha value for beta distribution')
    parser.add_argument('--dummynumber', default=0, type=int, help='number of dummy label.')
    parser.add_argument('--shmode', action='store_true')

    # dataset args
    parser.add_argument('--dataset_name', default="cognitive", type=str,
                        choices=["Constraint", "POLITIFACT", "LIAR-PLUS", "POLITIFACT", "cognitive"])
    parser.add_argument('--data_path', type=str, default='/path/code/TELLER_label/data/')
    parser.add_argument('--mode', type=str, default='multiple', choices=['binary', 'multiple'])
    # choose fewer smale for testing
    parser.add_argument('--num_eval_samples', default=5, type=int)
    parser.add_argument('--shot_number', default=0, type=int)
    parser.add_argument('--save_path', default="/path/code/TELLER_label/data/cognitive/report.json",
                        type=str)
    parser.add_argument('--save_all_path', default='/path/code/TELLER_label/data/save', type=str)

    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo",
                        choices=["flan-t5-xxl", "flan-t5-xl", "flan-t5-large", "flan-t5-base", "flan-t5-small",
                                 # flan有5种，lm2, gpt1
                                 "Llama-2-7b-chat-hf",
                                 "Llama-2-13b-chat-hf", "gpt-3.5-turbo"])
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--evi_flag', action="store_true")
    parser.add_argument('--eval_mode', type=str, default='logics', choices=['logics', 'sampling'])

    # the parameters of the logic model
    parser.add_argument('--num_conjuncts', default=5, type=int)
    parser.add_argument('--n_out', default=15, type=int, choices=[2, 15])  # 输出类别数
    parser.add_argument('--delta', default=1, type=float)  # 0.01 default
    parser.add_argument('--weight_init_type', default="normal", type=str, choices=["normal", "uniform"])
    parser.add_argument('--mask_flag', default=-2, type=int, choices=[-2, 0])
    parser.add_argument('--initial_delta', '-initial_delta', type=float, default=1,
                        help='initial delta.')

    parser.add_argument('--delta_decay_delay', '-delta_decay_delay', type=int, default=1,
                        help='delta_decay_delay.')

    parser.add_argument('--delta_decay_steps', '-delta_decay_steps', type=int, default=1,
                        help='delta_decay_steps.')
    # 0.01 1.3 -> 25 0.1 1.1
    parser.add_argument('--delta_decay_rate', '-delta_decay_rate', type=float, default=1.1,
                        help='delta_decay_rate.')
    # the logic model type
    parser.add_argument('--type_of_logic_model', default="logic", type=str,
                        choices=["gnn_logic", "gnn_logic_eo", "logic", "mlp", "tree", "bayes"])

    # the parameters of training the logic model， optimizer, schedule
    parser.add_argument('--SGD', '-sgd', action='store_true', help='use optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--n_steps_per_epoch', default=1, type=int)
    parser.add_argument('--scheduler', '-sch', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'CosLR'])
    parser.add_argument('--step_size', '-stp', type=int, default=20, help='fixed step size for StepLR')

    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--n_batch_step', type=int, default=10,
                        help='the number of batches per step for delta scheduler')
    parser.add_argument('--batchsize', default=32, type=int)


    parser.add_argument('--gqfile',
                        default="/path/code/BiasMind/data/cognitive/Qwen2.5-7B-Instruct_False_simplified_hand_compare_overcon_outcome_stereotype_5test_newscore_merges0711.json",
                        type=str)
    parser.add_argument('--evo_flag', action="store_true")
    parser.add_argument('--evo_file', default=None, type=str)

    parser.add_argument('--graph_flag', action="store_true")
    parser.add_argument('--graph_merge', default="u46", type=str,
                        choices=["tanh_adjustment", "u46", "absolute_s", "ref_dif", "ref_sem",
                                 "multiply"])

    parser.add_argument('--sample_u_flag', action="store_true")

    # loss type
    parser.add_argument('--type_of_loss', default="focal", type=str, choices=["focal", "ce"])
    parser.add_argument('--focal_alpha', default="alpha1", type=str,
                        choices=["alpha1", "alpha2", "alpha3", "alpha4"])
    parser.add_argument('--focal_gamma', default=2.0, type=float, choices=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
                                                                           5.0])
    # save the model
    # bestmodel_pruned  bestmodel
    parser.add_argument('--best_target_ckpoint', default="bestmodel", type=str)
    parser.add_argument('--best_dir', default="xx.pt", type=str)
    parser.add_argument('--save_flag', action="store_true")

    # the parameters of decision tree
    parser.add_argument('--max_depth', default=6, type=int, help='max_depth of decision tree')
    parser.add_argument('--max_leaf_nodes', default=30, type=int, help='max_leaf_nodes of decision tree')
    parser.add_argument('--min_weight_fraction_leaf', default=0.01, type=float,
                        help='min_weight_fraction_leaf of decision tree')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ############################# eval by LLMs
    args = parse_args()
    # predifine
    if args.evi_flag:
        gq_files = ["flan-t5-large_True.json", "flan-t5-xl_True.json", "flan-t5-xxl_True.json",
                    "Llama-2-7b-chat-hf_True.json",
                    "Llama-2-13b-chat-hf_True.json"]
        # gq_files = ["gpt-3.5-turbo_True.json"]
    else:
        gq_files = [
            "/path/code/BiasMind/data/cognitive/Qwen2.5-7B-Instruct_False_simplified_hand_compare_overcon_outcome_stereotype_groupin_9test_newscore_merges0717.json"]

    dir_best = {"cognitive": "群体内偏爱_0.4wiki_nos_val_acc_positive_new_score_lr0.001_decay0.0001_numcoj100_epoch_20_bs_128_gflag_False_gmerge_tanh_adjustment_loss_focal_alp_alpha1"}
    con_dict = {"cognitive": 100}
    lr_dict = {"cognitive": 0.001}
    wd_dict = {"cognitive": 0.0001}
    args.best_dir = dir_best[args.dataset_name]

    if args.n_out == 2:
        args.mode = 'binary'
    else:
        args.mode = 'multiple'
    wd = wd_dict[args.dataset_name]
    lr = lr_dict[args.dataset_name]
    conjunct = con_dict[args.dataset_name]

    final_results_wd_con = {}
    final_results = {}

    gq_file = "/path/code/BiasMind/data/cognitive/Qwen2.5-7B-Instruct_False_simplified_hand_compare_overcon_outcome_stereotype_groupin_9test_newscore_merges0717.json"


    args.num_conjuncts = conjunct
    args.weight_decay = wd
    args.gqfile = gq_file

    configure = []
    start = 1
    end = 106
    for i in range(start, end + 1):
        configure.append((f'P{i}', 1))
    print("=====configure.len====", len(configure))

    save_path = os.path.join(args.data_path, args.dataset_name, args.best_dir, args.best_target_ckpoint + ".pth")
    print("=========MODEL save_path:", save_path)



    state = torch.load(save_path)
    para = state['net']


    logic_model = GNN_DNF(num_conjuncts=conjunct, n_out=args.n_out, delta=state['delta'], configure=configure,
                          weight_init_type=args.weight_init_type, graph_flag=args.graph_flag,
                          graph_merge=args.graph_merge)



    logic_model.load_state_dict(para)
    logic_model = logic_model.to(args.device)

    e = Prune(dataset_name=args.dataset_name, mode=args.mode, data_path=args.data_path,
              gq_file=args.gqfile, sq_file=None, model_name=args.model_name, args=args)
    reported_test_metrics = e.post_processing(logic_model)
    print(reported_test_metrics)
