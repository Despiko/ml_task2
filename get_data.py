import pandas as pd
import re

data = pd.read_csv('jigsaw.csv')
<<<<<<< HEAD
=======

>>>>>>> f61a14ce51a32446931d4b9dd3a9d9dbe9e8ea8a
def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('[^A-Za-z]',' ',str(x)))
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub(r"https?://\S+|www\.\S+",'',str(x)))
    return text

data['comment_text'] = clean(data['comment_text'])

data.to_csv('cleaned.csv')
<<<<<<< HEAD

=======
>>>>>>> f61a14ce51a32446931d4b9dd3a9d9dbe9e8ea8a
