import pandas as pd
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay

df = pd.read_csv("DATASET/udemy_courses.csv")
df['course_title'] = df['course_title'].apply(nfx.remove_stopwords)
df['course_title'] = df['course_title'].apply(nfx.remove_special_characters).str.lower()

Xfeatures = df['course_title']
ylabels = df['subject']

tfidf_vec = TfidfVectorizer()

X = tfidf_vec.fit_transform(Xfeatures)
X.todense()
df_vec = pd.DataFrame(X.todense(), columns=tfidf_vec.get_feature_names_out())

x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)

lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)

print(lr_model.score(x_test, y_test))
y_pred = lr_model.predict(x_test)
cm=confusion_matrix(y_pred,y_test)
print(cm)
df['subject'].unique()
ConfusionMatrixDisplay(cm, display_labels=lr_model.classes_)
print(classification_report(y_pred,y_test))
ex = "Building a Simple ML Web App"

def vectorize_text(text):
    my_vec = tfidf_vec.transform([text])
    return my_vec.toarray()

vectorize_text(ex)
sample1 = vectorize_text(ex)
lr_model.predict(sample1)
lr_model.predict_proba(sample1)
lr_model.classes_

import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(lr_model, pickle_out)
pickle_out.close()

