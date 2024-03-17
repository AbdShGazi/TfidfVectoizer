#!/usr/bin/env python
# coding: utf-8

# In[8]:

#هذا الكود المستخدم لتنظيف البيانات
import pandas as pd
import re
from nltk.corpus import stopwords
     #الطريقة المستخدمة لتنظيف الملف
def clean_text(text):
    # ازالة اي شيء غير الحروف الابجديه العربية
    text = re.sub(r'[^ء-ي\s]', '', text) #ازالة اي شي لا يشمل الحرف الابجديه العربية
    # ازالة كلمات التوقف
    stop_words = set(stopwords.words('arabic'))
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# قراءة الملف قبل التحليل والتنظيف
df_uncleaned = pd.read_csv('NotCleanedcorprusB.csv')

# تنظيف الملف 
df_cleaned = pd.DataFrame({
    'x': df_uncleaned['x'].apply(clean_text),#تنظيف العمود الذي يحتوي على البيانات بناءا على الفنكشن المعرف كما فوق  
    'y': df_uncleaned['y']  
})
# حفظ الملف الجديد بعد التنظيف
df_cleaned.to_csv('cleaned_corpus.csv', index=False)
########################################################################################################################

# In[46]:
#The proram:
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
data = pd.read_csv("cleaned_corpus.csv")

# التقسيم الى قسم تدريب وقسم تجربة
X_train, X_test, y_train, y_test = train_test_split(data["x"], data["y"], test_size=0.3)
# CountVectorizer طريقة ال
count_vectorizer = CountVectorizer()
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)
# Naive bayes using countvectorizer
nb_classifier_count = MultinomialNB()
nb_classifier_count.fit(count_train, y_train)
pred_count = nb_classifier_count.predict(count_test)

# accuracy matrix and confusion المصفوفات
print("Accuracy with CountVectorizer:", metrics.accuracy_score(y_test, pred_count))
print("Confusion Matrix with CountVectorizer:")
print(metrics.confusion_matrix(y_test, pred_count))
print("=============End of Counting Part========")

# TFIDFطريقة ال
tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# naive bayes using tfidf
nb_classifier_tfidf = MultinomialNB()
nb_classifier_tfidf.fit(tfidf_train, y_train)
pred_tfidf = nb_classifier_tfidf.predict(tfidf_test)

# accuracy matrix and confusion المصفوفات
print("Accuracy with TFIDF:", metrics.accuracy_score(y_test, pred_tfidf))
print("Confusion Matrix with TFIDF:")
print(metrics.confusion_matrix(y_test, pred_tfidf))
print("=============End of TFIDF Part========")

# رسم المنحنى كما المطلوب
info_data = pd.read_csv("info.csv")
rolling_mean = info_data.rolling(window=3).mean()
plt.figure(figsize=(6, 6), dpi=100)
plt.title("Counting vs TFIDF")
plt.xlabel("Experiment", fontsize=10)
plt.ylabel("Accuracy", fontsize=10)
plt.grid()
plt.plot(rolling_mean["Case"], rolling_mean["Count"], label="CountVectorizer")
plt.plot(rolling_mean["Case"], rolling_mean["TFIDF"], label="TFIDF")
plt.legend()
plt.show()


# In[ ]:




