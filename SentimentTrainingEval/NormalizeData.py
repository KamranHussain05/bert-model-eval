
import pandas as pd

df = pd.read_csv("daily_weather_training_data.csv")
df['label'].value_counts()

# Equalize the number of labels
def equalizeData(df):
    neutral, positive, negative = [], [], []
    for i in range(len(df.label)):
        if df.label[i] == 2:
            neutral.append(df.iloc[i])
        elif df.label[i] == 1:
            positive.append(df.iloc[i])
        elif df.label[i] == 0:
            negative.append(df.iloc[i])
    return pd.DataFrame(columns=['text', 'label'], data=neutral[0:5922] + positive[0:5922] + negative[0:5922])

df = equalizeData(df)
print(df['label'].value_counts())
df.to_csv("daily_weather_training_data_normalized.csv", index=False, header=True, encoding='utf-8')
print("Done")