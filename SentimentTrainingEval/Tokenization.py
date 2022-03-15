
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import stanza as nlp
import torch
import json

import skipthoughts

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dict():
    with open('dict.json', 'r') as f:
        return json.load(f)

def main():
    stanza = nlp.Pipeline('en', processors='pos, tokenize, lemma', use_gpu=torch.cuda.is_available())
    data = str(input("Phrase: "))

    tokens = []
    for word in word_tokenize(data):
        if word not in stopwords.words('english'):
            tokens.append(word)

    cleaned = stanza(data)
    print(cleaned)
    print(tokens)


class SkipThoughtsVectorizer(object):
    def __init__(self, **kwargs):
        self.model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(self.model)

    def fit_transform(self, raw_documents, y):
        return self.encoder.encode(raw_documents, verbose=False)

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def transform(self, raw_documents, copy=True):
        return self.fit_transform(raw_documents, None)

feature_union = ('feature_union', FeatureUnion([
        ('skipthought', SkipThoughtsVectorizer()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ]))
    pipeline_both = Pipeline(steps=[feature_union,
                                    ('classifier', LogisticRegression())])

for train_size in (20, 50, 100, 200, 500, 1000, 2000, 3000, len(tweets_train)):
    print(train_size, '--------------------------------------')

classes_train[:train_size])
    # both
    pipeline_both.fit(tweets_train[:train_size], classes_train[:train_size])
    print('skipthought+tfidf', pipeline_both.score(tweets_test, classes_test))

if __name__ == '__main__':
    main()