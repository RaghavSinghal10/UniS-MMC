import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTFeatureExtractor, ViTModel, AutoProcessor, CLIPModel

# vit base model from https://huggingface.co/google/vit-base-patch16-224
# vit large model from https://huggingface.co/google/vit-large-patch16-224

class ImageEncoder_New(nn.Module):
    def __init__(self, pretrained_dir, image_encoder='base'):
        """
        image_encoder: base / large
        """
        super(ImageEncoder_New, self).__init__()

        self.image_encoder = image_encoder

        assert image_encoder in ['vit_base', 'vit_large', 'clip_vit_base']

        # directory is fine
        if image_encoder in ['vit_base']:
            tokenizer = ViTFeatureExtractor
            model = ViTModel
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/vit_base_224/')
            self.model = model.from_pretrained(pretrained_dir+'/vit_base_224/')
            # self.tokenizer = tokenizer.from_pretrained('google/vit-base-patch16-224')
            # self.model = model.from_pretrained('google/vit-base-patch16-224')
        elif image_encoder in ['vit_large']:
            tokenizer = ViTFeatureExtractor
            model = ViTModel
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/vit_large_224/')
            self.model = model.from_pretrained(pretrained_dir+'/vit_large_224/')
        elif image_encoder in ['clip_vit_base']:
            tokenizer = AutoProcessor
            model = CLIPModel
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/clip_vit_base/')
            self.model = model.from_pretrained(pretrained_dir+'/clip_vit_base/')            

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, pixel_values):
        """
        pixel_values:
        """

        if self.image_encoder in ['vit_base', 'vit_large']:
            last_hidden_state = self.model(pixel_values=pixel_values).last_hidden_state
        elif self.image_encoder in ['clip_vit_base']:
            outputs = self.model.vision_model(pixel_values=pixel_values)
            last_hidden_state = outputs.last_hidden_state
            # print("clip last hidden state", last_hidden_state.shape)
            

        return last_hidden_state
