import pandas as pd
import re
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess_review(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    text = ' '.join(text.split())
    
    tokens = word_tokenize(text)
    
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop]
    
    return " ".join(processed_tokens)

tqdm.pandas()

# 1. Загрузка набора данных
df = pd.read_csv('reviews.csv')

# 2. Подготовка данных для обучения
df['review'] = df['review'].progress_apply(preprocess_review)

df['label'] = df['sentiment'].progress_apply(lambda x: 1 if x == 'positive' else 0)

# 3. Сохранение обработанных данных в новый CSV-файл
df[['review', 'label']].to_csv('reviews_preprocessed.csv', index=False)
