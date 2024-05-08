import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import random
import os
import pandas as pd
import math
import json
import requests
import io
from urllib.request import Request, urlopen
import socket
import time 


def soft_clip_loss(preds, targs, temp=0.125):

    preds = F.normalize(preds, dim=-1)
    targs = F.normalize(targs, dim=-1)

    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss


def mixco_image(images, beta=0.15, s_thresh=0.5):

    perm = torch.randperm(images.shape[0]).to(images.device)
    images_shuffle = images[perm].to(images.device,dtype=images.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([images.shape[0]]).to(images.device,dtype=images.dtype)
    select = (torch.rand(images.shape[0]) <= s_thresh).to(images.device)
    betas_shape = [-1] + [1]*(len(images.shape)-1)
    images[select] = images[select] * betas[select].reshape(*betas_shape) + \
        images_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    
    return images, perm, betas, select

def mixco_text(text, beta=0.15, s_thresh=0.5):

    perm = torch.randperm(text.shape[0]).to(text.device)
    text_shuffle = text[perm].to(text.device,dtype=text.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([text.shape[0]]).to(text.device,dtype=text.dtype)
    select = (torch.rand(text.shape[0]) <= s_thresh).to(text.device)
    betas_shape = [-1] + [1]*(len(text.shape)-1)
    text[select] = text[select] * betas[select].reshape(*betas_shape) + \
        text_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    
    return text, perm, betas, select

def mixco_nce(mix_modality, fix_modality, temp=0.1, perm=None, betas=None, bidirectional=True):

    mix_modality = F.normalize(mix_modality, dim=-1)
    fix_modality = F.normalize(fix_modality, dim=-1)
    cos_sim = (mix_modality @ fix_modality.T)/temp

    probs = torch.diag(betas)
    probs[torch.arange(mix_modality.shape[0]).to(mix_modality.device), perm] = 1 - betas

    loss = -(cos_sim.log_softmax(-1) * probs).sum(-1).mean()

    if bidirectional:
        loss2 = -(cos_sim.T.log_softmax(-1) * probs.T).sum(-1).mean()
        loss = (loss + loss2)/2

    return loss


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def nce(image_embeddings, text_embeddings, temperature=0.1):

    # image_embeddings = F.normalize(image_embeddings, dim=-1)
    # text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Calculating the Loss
    logits = (text_embeddings @ image_embeddings.T) / temperature
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
        (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (images_loss + texts_loss) / 2.0

    return loss