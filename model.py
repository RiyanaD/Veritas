
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
train_df = pd.read_csv('train.csv')
print(f"The shape of the dataset is: {train_df.shape}")
print(train_df.head())
print(train_df.info())
train_df.fillna(" ", inplace= True)
X = train_df.drop(columns='label', axis=1)
y = train_df['label']

def apply_stemming(content, stemmer):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
stemmer = PorterStemmer()
train_df['text'] = train_df['text'].apply(lambda content: apply_stemming(content, stemmer))
print(train_df['text'].iloc[0])
