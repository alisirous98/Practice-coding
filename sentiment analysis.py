# import the packages
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

from transformers import pipeline

import torch

# import the model

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = input("Enter the text to analyze sentiment: ")

model_outputs = classifier(sentences)
print(model_outputs[0]) # produces a list of dicts for each of the labels

# to see what labels the dataset provides
for item in model_outputs[0]:
        print(item["label"])
# To see the score for a the specific label "sadness"
for item in model_outputs[0]:
    if item["label"] == "admiration":
        print(f"the admiration score for this content is {item['score']}")
        break
    
