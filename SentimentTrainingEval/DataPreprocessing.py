import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

start = time.time()
path = "daily_weather_2020.csv"

df = pd.read_csv(path, usecols=['summary', 'icon'], dtype=str, encoding='utf-8', low_memory=False)
df.head()

# Print the dataframe basic information
df = df.rename(columns={'summary':'text', 'icon': 'labels'})
df.info()


# Set up the validation column in the dataframe without modifying the raw data
def setValidation():
    print('VALIDATING DATA...')
    for i in range(len(df.labels)):
        val = df.labels[i]
        if val == 'rain' or val == 'snow' or val == 'wind' or val == 'fog':
            df.labels[i] = 'negative'  # negative sentiment
        elif val == 'clear-day':
            df.labels[i] = 'positive'  # positive sentiment
        else:
            df.labels[i] = 'neutral'   # neutral (possibly not enough data)


setValidation()
print(df['labels'].value_counts())


# Equalize the number of labels
def equalizeData(df):
    neutral, positive, negative = [], [], []
    for i in range(len(df.labels)):
        if df.labels[i] == 'neutral':
            neutral.append(df.iloc[i])
        elif df.labels[i] == 'positive':
            positive.append(df.iloc[i])
        elif df.labels[i] == 'negative':
            negative.append(df.iloc[i])
    return pd.DataFrame(columns=['text', 'labels'], data=neutral[0:8527] + positive[0:8527] + negative[0:8527])


balanced_df = equalizeData(df.copy())
balanced_df['labels'] = balanced_df['labels'].astype(str)
balanced_df.describe()
sns.countplot(balanced_df.labels)
print(balanced_df['labels'].value_counts())


# 0,1,2 : positive,negative,neutral
def making_label(st):
    if st == 'positive':
        return 1
    elif st == 'neutral':
        return 2
    else:
        return 0


balanced_df['labels'] = balanced_df['labels'].apply(making_label)
balanced_df.columns = ["text", "label"]

print(balanced_df['label'].value_counts())

balanced_df.to_csv('daily_weather_training_data.csv', index=False, header=True, encoding='utf-8')
print(" === Data Refactor Completed === ")
print(" === Total Time : ", time.time()-start, "seconds === ")