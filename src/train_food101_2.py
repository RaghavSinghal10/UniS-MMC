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

from model.model import MMC
from data.dataloader import MMDataLoader
from src.metrics import collect_metrics
from src.functions import save_checkpoint, load_checkpoint, dict_to_str, count_parameters

__all__ = ['TrainModule']

logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger('MMC')


# To decide the lr scheduler
def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


# To decide the optimizer
def get_optimizer(model, args):
    # if args.local_rank in [-1]:
    if args.mmc not in ['V']:
        text_enc_param = list(model.module.text_encoder.named_parameters())
        text_clf_param = list(model.module.text_classfier.parameters())
    if args.mmc not in ['T']:
        img_enc_param = list(model.module.image_encoder.parameters())
        img_clf_param = list(model.module.image_classfier.parameters())
    mm_clf_param = list(model.module.mm_classfier.parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if args.mmc in ['V']:
        optimizer_grouped_parameters = [
            {"params": img_enc_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_img_tfm},
            {"params": img_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_img_cls},
            {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
        ]
    elif args.mmc in ['T']:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in text_enc_param if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
            {"params": [p for n, p in text_enc_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
             'lr': args.lr_text_tfm},
            {"params": text_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},
            {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in text_enc_param if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
            {"params": [p for n, p in text_enc_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
             'lr': args.lr_text_tfm},
            {"params": text_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},
            {"params": img_enc_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_img_tfm},
            {"params": img_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_img_cls},
            {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
        ]
    optimizer = optim.Adam(optimizer_grouped_parameters)

    return optimizer


def train(args):

    train_loader, valid_loader, test_loader = MMDataLoader(args)
    data = train_loader, valid_loader, test_loader

    if args.local_rank in [-1]:
        model = DataParallel(MMC(args))
        model = model.to(args.device)
    else:
        model = MMC(args).to(args.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)
        
    if args.local_rank in [-1, 0]:
        print(f'\nThe model has {count_parameters(model)} trainable parameters')

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    model.train()
    train_loss_m = 0
    gradient_accumulation_steps = int(args.batch_gradient / args.batch_size)

    accuracy_test_history = []

    for epoch in range(args.num_epoch):
        train_loader, valid_loader, test_loader = data

        if args.local_rank not in [-1]:
            train_loader.sampler.set_epoch(epoch)

        for i, (batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label) in enumerate(train_loader):

                text = text_input_ids.to(args.device) 
                text_token_type_ids.to(args.device)
                text_attention_mask.to(args.device)
                image = batch_image.to(args.device)
                labels = batch_label.to(args.device).view(-1)

                # optimizer.zero_grad()
                loss, loss_m, logit_m = model(text, image, None, labels)
                # print(loss)
                loss = loss.sum() # / gradient_accumulation_steps
                loss.backward()

                exit()
                
                # optimizer.step()
                train_loss_m += loss_m.sum().item()
                wandb.log({"loss": train_loss_m})

                if (i+1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                running_loss += train_loss_m.detach().cpu().item()

        scheduler.step()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))


        model.eval()
        
        with torch.no_grad():

            y_pred = []
            y_true = []

            for i, (batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label) in enumerate(train_loader):
                    
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

if __name__ == '__main__':
    main()