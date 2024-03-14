import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from sklearn.metrics import accuracy_score
import argparse
import wandb
from model.model import MMC
from model.model_new import MMC_new
from data.dataloader import MMDataLoader
from src.metrics import collect_metrics
from src.functions import save_checkpoint, load_checkpoint, dict_to_str, count_parameters
from src.config import Config

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

__all__ = ['TrainModule']

torch.cuda.current_device()

# os.environ['TORCH_USE_CUDA_DSA'] = '1'

wandb.init(project='MMC')

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default='MMC', help='project name')
parser.add_argument('--dataset', type=str, default='Food101', help='support N24News/Food101')
parser.add_argument('--text_type', type=str, default='headline', help='support headline/caption/abstract')
parser.add_argument('--mmc', type=str, default='UniSMMC', help='support UniSMMC/UnSupMMC/SupMMC')
parser.add_argument('--mmc_tao', type=float, default=0.07, help='use supervised contrastive loss or not')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--lr_mm', type=float, default=1e-3, help='--lr_mm')
parser.add_argument('--min_epoch', type=int, default=1, help='min_epoch')    
parser.add_argument('--valid_step', type=int, default=50, help='valid_step')              
parser.add_argument('--max_length', type=int, default=512, help='max_length')

parser.add_argument('--text_encoder_1', type=str, default='deberta_base', help='bert_base/roberta_base/bert_large')
parser.add_argument('--text_encoder_2', type=str, default='roberta_base', help='bert_base/roberta_base/bert_large')
parser.add_argument('--image_encoder_1', type=str, default='vit_base', help='vit_base/vit_large')
parser.add_argument('--image_encoder_2', type=str, default='clip_vit_base', help='vit_base/vit_large')

parser.add_argument('--text_out', type=int, default=768, help='text_out')
parser.add_argument('--text_out_1', type=int, default=768, help='text_out')
parser.add_argument('--text_out_2', type=int, default=768, help='text_out')
parser.add_argument('--img_out', type=int, default=768, help='img_out')
parser.add_argument('--img_out_1', type=int, default=768, help='img_out')   
parser.add_argument('--img_out_2', type=int, default=768, help='img_out')    

parser.add_argument('--post_dim', type=int, default=256, help='post_dim')
parser.add_argument('--output_dim', type=int, default=101, help='output_dim')

parser.add_argument('--lr_text_tfm', type=float, default=5e-5, help='--lr_text_tfm')
parser.add_argument('--lr_img_tfm', type=float, default=5e-5, help='--lr_img_tfm')
parser.add_argument('--lr_text_cls', type=float, default=1e-4,help='--lr_text_cls')
parser.add_argument('--lr_img_cls', type=float, default=1e-4, help='--lr_img_cls')
parser.add_argument('--lr_mm_cls', type=float, default=1e-4, help='--lr_mm_cls')

parser.add_argument('--text_dropout', type=float, default=0.1, help='--text_dropout')
parser.add_argument('--img_dropout', type=float, default=0.1, help='--img_dropout')
parser.add_argument('--mm_dropout', type=float, default=0.0, help='--mm_dropout')

parser.add_argument('--nplot', type=str, default='', help='MTAV')
parser.add_argument('--data_dir', type=str, default='./datasets/', help='support wmsa') 
parser.add_argument('--test_only', type=bool, default=False,  help='train+test or test only')
parser.add_argument('--pretrained_dir', type=str, default='./pretrained_models', help='path to pretrained models from Hugging Face.')
parser.add_argument('--model_save_dir', type=str, default='Path/To/results/models', help='path to save model parameters.')
parser.add_argument('--res_save_dir', type=str, default='Path/To/results/results', help='path to save training results.')
parser.add_argument('--fig_save_dir', type=str, default='Path/To/results/imgs', help='path to save figures.')
parser.add_argument('--logs_dir', type=str, default='Path/To/results/logs', help='path to log results.')  # NO
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--seeds', nargs='+', type=int, help='set seeds for multiple runs!')
parser.add_argument('--model_path', type=str, default='./Path/To/results/models', help='path to load model parameters')
parser.add_argument('--save_model', type=bool, default=True, help='save model or not')


torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

args = parser.parse_args()
config = Config(args)
args = config.get_config()


if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")

args.device = device
args.data_dir = os.path.join(args.data_dir, args.dataset)

wandb.config.update(args)

print(args)



# To decide the lr scheduler
def get_scheduler(optimizer, args):
    # return optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor

    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch)

# To decide the optimizer
def get_optimizer(model, args):

    text_enc_1_param = list(model.module.text_encoder_1.named_parameters())
    text_enc_2_param = list(model.module.text_encoder_2.named_parameters())
    text_combine_param = list(model.module.text_combine.parameters())
    text_clf_param = list(model.module.text_classfier.parameters())


    # a = list(model.module.image_encoder_1.parameters())
    # print("model.module.named_paramters value: " ,a[0] )  
    # print("model.module.named_paramters length: " , len(a) ) #200
    # print("model.module.named_paramters type: " ,type(a)) #list
    # print("model.module.named_paramters type of one particular element: " ,type(a[0])) #torch.nn.paramater.Parameter

    img_enc_1_param = list(model.module.image_encoder_1.parameters())
    img_enc_2_param = list(model.module.image_encoder_2.parameters())
    img_combine_param = list(model.module.image_combine.parameters())
    img_clf_param = list(model.module.image_classfier.parameters())

    mm_clf_param = list(model.module.mm_classfier.parameters())

    # print("image encoder: " ,type(img_enc_1_param[1]))
    # print("image classfication: ", type(img_clf_param[1]))   
    # print("text encoder: ", type(text_enc_1_param[1]))
    # print("text classification: ", type(text_clf_param[1]))
    # print("mm classification: " , type(mm_clf_param[1]))
    # print("combine param classification:" , type(text_combine_param[1]))
    # print("image combine param: ", type(img_combine_param[1]))
    
    # exit()

    # print("text enc 1 params:", type(text_enc_1_param))
    # print("text enc 2 params:", type(text_enc_2_param))
    # print("text combine params:", type(text_combine_param))
    # print("text clf params:", type(text_clf_param))

    # print("img enc 1 params:", type(img_enc_1_param))
    # print("img enc 2 params:", type(img_enc_2_param))
    # print("img combine params:", type(img_combine_param))
    # print("img clf params:", type(img_clf_param))

    # print("mm clf params:", type(mm_clf_param))

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

 
    optimizer_grouped_parameters = [

        {"params": [p for n, p in text_enc_1_param if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
        {"params": [p for n, p in text_enc_1_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
            'lr': args.lr_text_tfm},
        {"params": [p for n, p in text_enc_2_param if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
        {"params": [p for n, p in text_enc_2_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
            'lr': args.lr_text_tfm},
        {"params": text_combine_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_tfm},
        {"params": text_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},

        {"params": img_enc_1_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_img_tfm},
        {"params": img_enc_2_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_img_tfm},
        {"params": img_combine_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_img_tfm},
        {"params": img_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_img_cls},

        {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
    ]

    optimizer = optim.Adam(optimizer_grouped_parameters)

    return optimizer


def main():

    train_loader, valid_loader, test_loader = MMDataLoader(args)

    # print number of train, valid, test samples
    print(f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")

    if args.local_rank in [-1]:
        model = DataParallel(MMC_new(args))
        model = model.to(args.device)
    else:
        model = MMC_new(args).to(args.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)
            
        
    if args.local_rank in [-1, 0]:
        print(f'\nThe model has {count_parameters(model)} trainable parameters')

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    train_loss_m = 0
    gradient_accumulation_steps = int(args.batch_gradient / args.batch_size)

    accuracy_test_history = []

    for epoch in range(args.num_epoch):
        
        running_loss = 0.0
        model.train()

        if args.local_rank not in [-1]:
            train_loader.sampler.set_epoch(epoch)

        for i, (batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label) in enumerate(train_loader):

                text = text_input_ids.to(args.device), text_token_type_ids.to(args.device), text_attention_mask.to(args.device)
                image = batch_image.to(args.device)
                labels = batch_label.to(args.device).view(-1)

                # print(text.shape, image.shape, labels.shape)
                # exit()
                # print((i+1), (i+1)%gradient_accumulation_steps)
                # optimizer.zero_grad()
                loss, loss_m, logit_m = model(text, image, None, labels)
                # print(loss)
                loss = loss.sum() # / gradient_accumulation_steps
                loss.backward()
                
                # optimizer.step()
                # train_loss_m += loss_m.sum().item()

                if (i+1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                running_loss += loss.detach().cpu().item()

                wandb.log({"loss": loss})


        scheduler.step()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))


        model.eval()
        
        with torch.no_grad():

            y_pred = []
            y_true = []

            for i, (batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label) in enumerate(test_loader):
                    
                text = text_input_ids.cuda(), text_token_type_ids.cuda(), text_attention_mask.cuda()
                image = batch_image.cuda()
                logit = model.module.infer(text, image, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())

            logits = torch.cat(y_pred)
            true = torch.cat(y_true).data.cpu().numpy()
            prob = F.softmax(logits, dim=1).data.cpu().numpy()

            results = collect_metrics(args.dataset, true, prob)
            # get acc from results
            accuracy_test = results['acc']
            accuracy_test_history.append(accuracy_test)

            max_accuracy = max(accuracy_test_history)

            wandb.log({"test_acc": accuracy_test, "max_test_acc": max_accuracy})

            print(f'Accuracy: {accuracy_test} | Max Accuracy: {max_accuracy}')

    if args.save_model:
        save_checkpoint(model, args.best_model_save_path)

if __name__ == '__main__':
    main()