from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd

import torch


class Sentiment_model:

    def __init__(self):
        # download label mapping
        task = 'sentiment'
        self.labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        self.labels = [row[1] for row in csvreader if len(row) > 1]

        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.tokenizer.model_max_length = 1071

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest")

        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def review_sentiment(self, review: str):
        encoded_input = self.tokenizer(review, return_tensors='pt', truncation=True, max_length=512)

        # Move the input tensor to GPU if available
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')

        output = self.model(**encoded_input)
        scores = output.logits.detach().cpu().numpy()[0]
        scores = np.exp(scores) / np.sum(np.exp(scores))
        ranking = np.argsort(scores)[::-1]
        ans = [(self.labels[i], scores[i]) for i in ranking]
        return ans

    # Define a function to apply review_sentiment to each row
    def apply_review_sentiment(self, row):
        review = row['reviews']
        sentiment_scores = self.review_sentiment(review)
        for sentiment, score in sentiment_scores:
            row[sentiment] = score
        return row
