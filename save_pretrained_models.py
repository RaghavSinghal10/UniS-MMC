# from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, ViTModel

# models = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
# tokenizers = [BertTokenizer, RobertaTokenizer]
# model_classes = [BertModel, RobertaModel]

# for model_name in models:
#     for tokenizer, model_class in zip(tokenizers, model_classes):
#         try:
#             model = model_class.from_pretrained(model_name)
#             tok = tokenizer.from_pretrained(model_name)
#             model.save_pretrained(f"./pretrained_models/{model_name}")
#             tok.save_pretrained(f"./pretrained_models/{model_name}")
#             print(f"Downloaded {model_name}")
#             break
#         except Exception as e:
#             continue

# For ViT model
# from transformers import ViTFeatureExtractor
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')
# model = ViTModel.from_pretrained('google/vit-large-patch16-224')
# model.save_pretrained("./pretrained_models/vit-large")
# feature_extractor.save_pretrained("./pretrained_models/vit-large")

# # For CLIP model
# from transformers import AutoProcessor, CLIPModel
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# model.save_pretrained("./pretrained_models/clip-vit-base")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# processor.save_pretrained("./pretrained_models/clip-vit-base")

# For Deberta model
# from transformers import DebertaTokenizer, DebertaModel
# tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
# model = DebertaModel.from_pretrained("microsoft/deberta-base")
# model.save_pretrained("./pretrained_models/deberta_base")
# tokenizer.save_pretrained("./pretrained_models/deberta_base")

# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F
# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
# model = AutoModel.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
# model.save_pretrained("./pretrained_models/multi-qa-mpnet-base-dot-v1")
# tokenizer.save_pretrained("./pretrained_models/multi-qa-mpnet-base-dot-v1")
