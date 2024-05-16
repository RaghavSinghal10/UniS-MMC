import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from sklearn.metrics import f1_score, accuracy_score
import argparse
import wandb
from model.utils import *
from model.model import MMC
from model.model_med import MMC_Med
from model.model_mmd import MMDynamic
from data.dataloader import MMDataLoader
from src.metrics import collect_metrics
from src.functions import save_checkpoint, load_checkpoint, dict_to_str, count_parameters
from src.config import Config
import time
from torch import optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

__all__ = ['TrainModule']

wandb.init(project='MMC-med')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='rosmap', help='support N24News/Food101')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')                                                    
parser.add_argument('--seeds', nargs='+', type=int, help='set seeds for multiple runs!')
parser.add_argument('--model_path', type=str, default='/raid/nlp/rajak/Multimodal/UniS-MMC/datasets', help='path to load model parameters')
parser.add_argument('--data_path', type=str, default='/raid/nlp/rajak/Multimodal/UniS-MMC/datasets', help='path to load data')
parser.add_argument('--save_model', type=bool, default=True, help='save model or not')
parser.add_argument('--test_only', action='store_true', help='test only or not')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--seed', type=int, default=69, help='seed')

parser.add_argument('--cross_attention', action='store_true', help='cross attention or not')
parser.add_argument('--image_embedding_mixup', action='store_true', help='image embedding mixup or not')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha for mixup')

parser.add_argument('--multi_mixup', action='store_true', help='multi mixco or not')
parser.add_argument('--mixup_pct', type=float, default=0.33, help='mixup percentage')
parser.add_argument('--lambda_mixup', type=float, default=0.1, help='lambda for mixup')
parser.add_argument('--mixup_beta', type=float, default=0.15, help='beta for mixup')
parser.add_argument('--mixup_s_thresh', type=float, default=0.5, help='s_thresh for mixup')

parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='scheduler')
parser.add_argument('--shuffle',action='store_true', help='shuffle or not')
parser.add_argument('--no_uni_pred', action='store_true', help='no_uni_pred or not')


torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

args = parser.parse_args()
config = Config(args)
args = config.get_config()

wandb.config.update(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)    
    return y_onehot

def prepare_trte_data(data_folder):
    num_view = 3
    print(data_folder)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view+1):
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    
    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        data_tensor_list[i] = data_tensor_list[i].to(args.device)
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels

def get_optimizer(model, args):
    
    mm_1_param = list(model.module.fe_1.parameters())
    mm_2_param = list(model.fe_2.parameters())
    mm_3_param = list(model.fe_3.parameters())

    mm_clf_1_param = list(model.cl_1.parameters())
    mm_clf_2_param = list(model.cl_2.parameters())
    mm_clf_3_param = list(model.cl_3.parameters())

    mm_clf_param = list(model.mm_classfier.parameters())

    optimizer_grouped_parameters = [
        {"params": mm_1_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_img_tfm},
        {"params": mm_2_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
        {"params": mm_3_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
        {"params": mm_clf_1_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_img_cls},
        {"params": mm_clf_2_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},
        {"params": mm_clf_3_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},
        {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
    ]
    
    optimizer = optim.Adam(optimizer_grouped_parameters)

    return optimizer

def get_scheduler(optimizer, args):
    if args.lr_scheduler == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
        )
    elif args.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch)
    
def train_epoch(data_list, label, model, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    #loss, _ = model(data_list, label)
    
    if epoch < int(args.mixup_pct * args.num_epoch):
        loss, loss_m, logit = model(data_list, label, use_soft_clip=False)
        loss = torch.mean(loss)
    else:
        loss, loss_m, logit = model(data_list, label, use_soft_clip=True)
        loss = torch.mean(loss)

    loss.backward()

    wandb.log({"final ce loss": loss_m.sum().detach().cpu().item()})
    wandb.log({"training loss": loss.detach().cpu().item()})
    # for param in model.parameters():
    #     print(param.grad)
    # exit()
    optimizer.step()
    # for param in model.parameters():
    #     print(param.grad)
    #     break

def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        #logit = model.infer(data_list)
        logit = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob

def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)

def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)


def main():

    test_inverval = 1
    
    if args.dataset == "brca":
        data_folder = os.path.join(args.data_path, "BRCA")
        hidden_dim = [500]
        #num_epoch = 2500
        lr = 1e-4
        step_size = 500
        num_class = 5
    
    elif args.dataset == "rosmap":
        data_folder = os.path.join(args.data_path, "ROSMAP")
        hidden_dim = [300]
        #num_epoch = 1000
        lr = 1e-4
        step_size = 500
        num_class = 2

    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_tr_tensor = labels_tr_tensor.cuda()
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
    dim_list = [x.shape[1] for x in data_tr_list]

    # model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5).to(args.device)
    # #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model = MMC_Med(args, input_dim_list=dim_list).to(args.device)
    #model = MMC(args)
    # print modules in the model
    # for name, module in model.named_modules():
    #     print(name)

    # exit()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_mm, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)

    #scheduler = get_scheduler(optimizer, args)
    #optimizer = get_optimizer(model, args)

    print("\nTraining...")
    test_acc_history = []
    test_f1_history = []
    test_auc_history = []
    test_f1_weighted_history = []
    test_f1_macro_history = []
    
    for epoch in range(args.num_epoch):
        train_epoch(data_tr_list, labels_tr_tensor, model, optimizer, epoch)
        scheduler.step()

        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_test_list, model)
            print("\nTest: Epoch {:d}".format(epoch))
            if args.dataset == "rosmap":

                test_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                test_f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                test_auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])

                test_acc_history.append(test_acc)
                test_f1_history.append(test_f1)
                test_auc_history.append(test_auc)

                max_test_acc = max(test_acc_history)
                max_test_f1 = max(test_f1_history)
                max_test_auc = max(test_auc_history)

                wandb.log({"test acc": test_acc})
                wandb.log({"test f1": test_f1})
                wandb.log({"test auc": test_auc})

                wandb.log({"max test acc": max_test_acc})
                wandb.log({"max test f1": max_test_f1})
                wandb.log({"max test auc": max_test_auc})

                print("Test ACC: {:.5f}".format(test_acc))
                print("Test F1: {:.5f}".format(test_f1))
                print("Test AUC: {:.5f}".format(test_auc))
                print("Max Test ACC: {:.5f}".format(max_test_acc))
                print("Max Test F1: {:.5f}".format(max_test_f1))
                print("Max Test AUC: {:.5f}".format(max_test_auc))

            else:

                test_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                test_f1_weighted = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                test_f1_macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')

                test_acc_history.append(test_acc)
                test_f1_weighted_history.append(test_f1_weighted)
                test_f1_macro_history.append(test_f1_macro)

                max_test_acc = max(test_acc_history)
                max_test_f1_weighted = max(test_f1_weighted_history)
                max_test_f1_macro = max(test_f1_macro_history)

                wandb.log({"test acc": test_acc})
                wandb.log({"test f1 weighted": test_f1_weighted})
                wandb.log({"test f1 macro": test_f1_macro})

                wandb.log({"max test acc": max_test_acc})
                wandb.log({"max test f1 weighted": max_test_f1_weighted})
                wandb.log({"max test f1 macro": max_test_f1_macro})

                print("Test ACC: {:.5f}".format(test_acc))
                print("Test F1 weighted: {:.5f}".format(test_f1_weighted))
                print("Test F1 macro: {:.5f}".format(test_f1_macro))
                print("Max Test ACC: {:.5f}".format(max_test_acc))
                print("Max Test F1 weighted: {:.5f}".format(max_test_f1_weighted))
                print("Max Test F1 macro: {:.5f}".format(max_test_f1_macro))
        #save_checkpoint(model.state_dict(), os.path.join(modelpath, data_folder))

if __name__ == '__main__':
    main()