

import fasttext
import pandas as pd

# Create the training text file
df = pd.read_csv('/Users/kamranhussain/Documents/GitHub/bert-model-eval/SentimentTrainingEval/daily_weather_training_data.csv', usecols=['text'])
df['text'].to_csv('training_text.txt', index=False, header=False, sep='\n')


model = fasttext.train_unsupervised(input="training_text.txt", model="cbow", epoch=50, dim=256)
model.save_model("fasttexttokenizer.bin")

print(model.get_word_vector('clear weather throughout the day'))
