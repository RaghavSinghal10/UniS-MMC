from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

# Get the image features
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = '/raid/nlp/rajak/eye-traking/ScanPath/hallucinated_RoI/hallucinated_RoI/t1_p01.png'
image = Image.open(url)

inputs = processor(images=image, return_tensors="pt")

image_features1 = model.get_image_features(**inputs)

print(image_features1.shape) # output shape of image features

from transformers import AutoTokenizer, DebertaModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaModel.from_pretrained("microsoft/deberta-base")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state