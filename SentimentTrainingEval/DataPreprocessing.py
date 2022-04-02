import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

path = "daily_weather_2020.csv"

df = pd.read_csv(path, usecols=['summary', 'icon'], dtype=str, encoding='utf-8', low_memory=True)
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
            df.labels[i] = 'negative' # negative sentiment
        elif val == 'clear-day':
            df.labels[i] = 'positive' # positive sentiment
        elif val == 'partly-cloudy-day' or val == 'cloudy':
            df.labels[i] = 'neutral'  # neutral (possibly not enough data)


# check the validation edit worked
def checkValidationSetting():
    print("CHECKING VALIDATION DATA")
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for i in range(len(df.labels)):
        print('Checking data refactor...')
        if df.labels[i] == 'positive':
            positive_count +=1
        elif df.labels[i] == 'negative':
            negative_count +=1
        elif df.labels[i] == 'neutral':
            neutral_count +=1

    print('positive Count: ' + str(positive_count))
    print('Unfavorable Count: ' + str(negative_count))
    print('Neutral Count: ' + str(neutral_count))


start = time.time()
setValidation()
df['labels'].value_counts()
print('==PROCESS COMPLETED==\n', time.time()-start, 'seconds')

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
    return pd.DataFrame(columns=['text', 'labels'], data=neutral[:8527] + positive[:8527] + negative[:8527])


df = equalizeData(df)
df['labels'] = df['labels'].astype(str)
df.describe()
df.info()
sns.countplot(df.labels)
df['labels'].value_counts()


# 0,1,2 : positive,negative,neutral
def making_label(st):
    if(st=='positive'):
        return 1
    elif(st=='neutral'):
        return 2
    else:
        return 0

df['labels'] = df['labels'].apply(making_label)
print(df.shape)

df.columns = ["text", "labels"]

df.to_csv('daily_weather_training_data.csv',index=True, header=True, encoding='utf-8')

print("Data Refactor Completed")