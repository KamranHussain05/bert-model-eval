
from nltk.tokenize import word_tokenize
data = str(input("Phrase: "))

tokens = word_tokenize(data.decode('utf-8'))
print(tokens)