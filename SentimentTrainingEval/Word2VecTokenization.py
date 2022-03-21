
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import stanza

print("Loading data...")
path = 'daily_weather_2020.csv'
df = pd.read_csv(path, usecols=['summary', 'icon'], low_memory=True, encoding='utf-8', dtype='str')
df.columns = ['text', 'labels']


# Refactor the classes into the labels
def refactorLabels():
    print("Refactoring Data...")
    for i in range(len(df.labels)):
        val = df.labels[i]
        if val == 'rain' or val == 'snow' or val == 'wind' or val == 'fog':
            df.labels[i] = 'negative'  # negative sentiment
        elif val == 'clear-day':
            df.labels[i] = 'positive'  # positive sentiment
        elif val == 'partly-cloudy-day' or val == 'cloudy':
            df.labels[i] = 'neutral'  # neutral (possibly not enough data)


# Equalize the number of labels
def equalizeData():
    print("Normalizing Data...")
    neutral, positive, negative = [], [], []
    for i in range(len(df.labels)):
        if df.labels[i] == 'neutral':
            neutral.append(df.iloc[i])
        elif df.labels[i] == 'positive':
            positive.append(df.iloc[i])
        elif df.labels[i] == 'negative':
            negative.append(df.iloc[i])
    return DataFrame(columns=['text', 'labels'], data=neutral[:8527] + positive[:8527] + negative[:8527])


refactorLabels()
df = equalizeData()

# Visualize the labels for human verification
print("Visualizing Data...")
sns.countplot(df.labels)
plt.show()
print(df['labels'].value_counts())

# Setup the Stanza pipeline and tokenize and lemmatize the data
nlp = stanza.Pipeline('en', processors='tokenize, lemma', use_gpu=True)

# Setup the tensorflow pipeline and models for word2vec, not using pretrains
