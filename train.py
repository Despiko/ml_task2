import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

from nltk.corpus import stopwords
import json



clean_data = pd.read_csv('cleaned.csv')

def lemmatize(text):
    m = WordNetLemmatizer()
    lemm_list = m.lemmatize(text)
    lemm_text = "".join(lemm_list)
    return lemm_text

corpus = list(clean_data['comment_text'].apply(lambda x: lemmatize(x)))


nltk.download('stopwords')

stopwords = set(stopwords.words('english'))


count_tf_idf = TfidfVectorizer(stop_words = stopwords)
tf_idf = count_tf_idf.fit_transform(corpus)

features = tf_idf
target = clean_data['toxic'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

model_reg = LogisticRegression(class_weight = 'balanced', C=0.5)

model_reg.fit(X_train, y_train)

predict_linear = model_reg.predict(X_test)


roc_auc = roc_auc_score(y_test, predict_linear)

f1_lin = f1_score(y_test, predict_linear)
with open("metrics.json", 'w') as outfile:
    json.dump({ "F1": f1_lin, "roc_auc": roc_auc}, outfile)

with open("report.txt", "w") as file:
    print('F1 логистической регрессии: {:.4f}'.format(f1_lin), file=file)
    print('',file=file)
    print('Матрица ошибок', file=file)
    print(confusion_matrix(y_test, predict_linear), file=file)
    print('',file=file)

roc_auc_score(y_test, predict_linear)

f1_lin = f1_score(y_test, predict_linear)
print('F1 логистической регрессии: {:.4f}'.format(f1_lin))
print()
print('Матрица ошибок')
print(confusion_matrix(y_test, predict_linear))
print()
