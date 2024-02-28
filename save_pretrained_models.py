from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, ViTModel

models = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
tokenizers = [BertTokenizer, RobertaTokenizer]
model_classes = [BertModel, RobertaModel]

for model_name in models:
    for tokenizer, model_class in zip(tokenizers, model_classes):
        try:
            model = model_class.from_pretrained(model_name)
            tok = tokenizer.from_pretrained(model_name)
            model.save_pretrained(f"./pretrained_models/{model_name}")
            tok.save_pretrained(f"./pretrained_models/{model_name}")
            print(f"Downloaded {model_name}")
            break
        except Exception as e:
            continue

# For ViT model
from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
model.save_pretrained("./pretrained_models/vit-base")
feature_extractor.save_pretrained("./pretrained_models/vit-base")