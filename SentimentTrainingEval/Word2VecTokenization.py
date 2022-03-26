import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import stanza

# Progress bae
# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='Completed', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print('\r')

print("Loading data...")
path = 'daily_weather_2020.csv'
df = pd.read_csv(path, usecols=['summary', 'icon'], low_memory=True, encoding='utf-8', dtype='str')
df.columns = ['text', 'labels']


# Refactor the classes into the labels
def refactorLabels():
    for i in range(len(df.labels)):
        printProgressBar(i, len(df), prefix='Refactoring Labels')
        val = df.labels[i]
        if val == 'rain' or val == 'snow' or val == 'wind' or val == 'fog':
            df.labels[i] = 'negative'  # negative sentiment
        elif val == 'clear-day':
            df.labels[i] = 'positive'  # positive sentiment
        elif val == 'partly-cloudy-day' or val == 'cloudy':
            df.labels[i] = 'neutral'  # neutral (possibly not enough data)


# Equalize the number of labels
def equalizeData(df):
    neutral, positive, negative = [], [], []
    for i in range(len(df.labels)):
        printProgressBar(i, len(df), prefix='Normalizing Data')
        if df.labels[i] == 'neutral':
            neutral.append(df.iloc[i])
        elif df.labels[i] == 'positive':
            positive.append(df.iloc[i])
        elif df.labels[i] == 'negative':
            negative.append(df.iloc[i])
    return DataFrame(columns=['text', 'labels'], data=neutral[:8527] + positive[:8527] + negative[:8527])


refactorLabels()
df = equalizeData(df)


# Visualize the labels for human verification
print("Visualizing Data...")
sns.countplot(df.labels)
plt.show()
print(df['labels'].value_counts())


# Setup the Stanza pipeline and tokenize and lemmatize the data
nlp = stanza.Pipeline('en', processors='pos, tokenize, lemma', use_gpu=True)

# Lemmatize the Text and create tokens
def LemmatizeAndSplit(phrase):
    lemmas = []
    doc = nlp(str(phrase))
    for i in range(len(doc.sentences)):
        for word in doc.sentences[i].words:
            lemmas.append(word.lemma)
    return lemmas


# Show the lemmatized and split sentence
lemmatized = LemmatizeAndSplit(df.text[0])
print(lemmatized)
print(len(df))
tokens = df
print(len(tokens))
for i in range(len(df)):
    df.text[i] = LemmatizeAndSplit(df.text[i])
    printProgressBar(i, len(df), prefix="Split and Lemmatize Data")



# Setup the tensorflow pipeline and models for word2vec, not using pretrains


