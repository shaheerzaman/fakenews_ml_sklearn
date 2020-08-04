import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import  LogisticRegression

df = pd.read_csv('data.csv')

df.loc[df['label'] == 'Fake', 'label'] = 'FAKE'
df.loc[df['label'] == 'fake', 'label'] = 'FAKE'
df.loc[df['source']=='facebook', 'source'] = 'Facebook'
df.text.fillna(df.title, inplace=True)

df.loc[5, 'label'] = 'FAKE'
df.loc[15, 'label'] = 'TRUE'
df.loc[43, 'label'] = 'FAKE'
df.loc[131, 'label'] = 'TRUE'
df.loc[242, 'label'] = 'FAKE'

df = df.sample(frac=1).reset_index(drop=True)
df.title.fillna('missing', inplace=True)
df.source.fillna('missing', inplace=True)

df['title_text'] = df['title'] + ' ' + df['text']

def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

df['title_text'] = df['title_text'].apply(preprocessor)
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tfidf = TfidfVectorizer(
    strip_accents=None, 
    lowercase=False, 
    preprocessor=None, 
    tokenizer=tokenizer_porter, 
    use_idf=True, 
    norm='l2', 
    smooth_idf=True, 
)

print(df['label'].value_counts())

print(df.head())

x = tfidf.fit_transform(df['title_text'])
y = df.label.values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=0, test_size=0.5, shuffle=False
)

clf = LogisticRegressionCV(cv=5, scoring='accuracy',
                    random_state=0, n_jobs=-1, verbose=3, max_iter=300)
                    .fit(x_train, y_train)

fake_news_model = open('fake_news_model.sav', 'wb')
pickle.dump(clf, fake_news_model)
fake_news_model.close()

filename = 'fake_news_model.sav'
saved_clf = picle.load(open(filename, 'rb'))
saved_clf.score(x_test, y_test)

from sklearn.metrics import classification_report, accuracy_score
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

                    







