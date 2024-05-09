import functools
import json
import os
from collections import Counter

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs

if __name__ == "__main__":
    path = "/raid/nlp/rajak/Multimodal/UniS-MMC/datasets/mmimdb/train.jsonl"
    labels, freqs = get_labels_and_frequencies(path)

    labels_imdb = {
    'Action': 0,
    'Adventure': 1,
    'Animation': 2,
    'Biography': 3,
    'Comedy': 4,
    'Crime': 5,
    'Documentary': 6,
    'Drama': 7,
    'Family': 8,
    'Fantasy': 9,
    'Film-Noir': 10,
    'History': 11,
    'Horror': 12,
    'Music': 13,
    'Musical': 14,
    'Mystery': 15,
    'Romance': 16,
    'Sci-Fi': 17,
    'Short': 18,
    'Sport': 19,
    'Thriller': 20,
    'War': 21,
    'Western': 22
    }

    freqs = [freqs[l] for l in labels_imdb.keys()]
    
    print(labels)
    print(freqs)


# [2154, 1609, 586, 772, 5105, 2287, 1194, 8414, 975, 1162, 202, 663, 1603, 632, 503, 1231, 3226, 1212, 281, 379, 3110, 804, 423]