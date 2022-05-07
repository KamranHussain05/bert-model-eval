import pandas as pd
import joblib
from SkipThoughtsMaster import skipthoughts
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# define skip-thoughts vectorizer class for scikit-learn
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


# Read the file
df = pd.read_csv('../SentimentTrainingEval/daily_weather_2020.csv', usecols=['summary', 'icon'], low_memory=True, dtype=str, encoding="utf-8")
df = df.rename(columns={'summary':'text', 'icon': 'labels'})

# Set up the validation column in the dataframe without modifying the raw data
print("SETTING UP DATA REFACTORING")
for i in range(len(df.labels)):
    val = df.labels[i]
    if val == 'rain' or val == 'snow' or val == 'wind' or val == 'fog':
        df.labels[i] = 'negative' # negative sentiment
    elif val == 'clear-day':
        df.labels[i] = 'positive' # positive sentiment
    elif val == 'partly-cloudy-day' or val == 'cloudy':
        df.labels[i] = 'neutral' # neutral sentiment
print("DATA REFACTORED")

# split the data into test and training sets
phrase_train,phrase_eval = train_test_split(df.text.astype(str),test_size=0.1)
classes_train, classes_eval = train_test_split(df.labels.astype(str),test_size=0.1)

# Convert the dataframes to lists
phrase_train = phrase_train.tolist()
phrase_eval = phrase_eval.tolist()
classes_train = classes_train.tolist()
classes_eval = classes_eval.tolist()


# Define pipelines for skipthought
pipeline_skipthought = Pipeline(steps=[('vectorizer', SkipThoughtsVectorizer()),
                        ('classifier', LogisticRegression())])

# Define pipeline for TF-IDF
pipeline_tfidf = Pipeline(steps=[('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
                        ('classifier', LogisticRegression())])

# Define the feature unions of the two pipelines
feature_union = ('feature_union', FeatureUnion([
    ('skipthought', SkipThoughtsVectorizer()),
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
]))

# Define the combined skip-thought and TF-IDF pipeline
pipeline_both = Pipeline(steps=[feature_union,
                        ('classifier', LogisticRegression(max_iter=1000000000))])

# Train the pipelines on the data and create vectors of the data
print("TRAINING PIPELINES")
for train_size in (20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000, 15000, len(phrase_train)):
    print(train_size, '--------------------------------------')
    # skipthought
    pipeline_skipthought.fit(phrase_train[:train_size], classes_train[:train_size])
    print ('skipthought', pipeline_skipthought.score(phrase_eval, classes_eval))

    # tfidf
    pipeline_tfidf.fit(phrase_train[:train_size], classes_train[:train_size])
    print('tfidf', pipeline_tfidf.score(phrase_eval, classes_eval))

    # both
    pipeline_both.fit(phrase_train[:train_size], classes_train[:train_size])
    print('skipthought+tfidf', pipeline_both.score(phrase_eval, classes_eval))


# Save the model with the trained Weights and vectors
print("SAVING MODEL")
joblib.dump(pipeline_both, 'skipthought-IFIDF_model.pkl')
