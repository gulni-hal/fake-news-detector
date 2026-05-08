import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# NLTK Stopwords'leri indir (sadece ilk çalıştırışta gerekir)
nltk.download('stopwords')

print("1. Veri setleri yükleniyor...")
# Verileri yükle
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Etiketleri ekle (Sahte = 0, Gerçek = 1)
df_fake["label"] = 0
df_true["label"] = 1

# İki veri setini birleştir ve karıştır
df_marge = pd.concat([df_fake, df_true], axis=0)
df = df_marge.sample(frac=1).reset_index(drop=True)

# Sadece 'text' (haber metni) ve 'label' sütunlarını tutalım
df = df[['text', 'label']]

print("2. Metin ön işleme (NLP) yapılıyor...")
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['text'] = df['text'].apply(wordopt)

# Bağımsız değişken (X) ve bağımlı değişken (y)
X = df['text']
y = df['label']

print("3. Eğitim ve test verisi ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("4. TF-IDF Vektörizasyon işlemi...")
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(X_test)

print("5. Model eğitiliyor (Logistic Regression)...")
LR = LogisticRegression()
LR.fit(xv_train, y_train)

# Test verisi ile modeli değerlendir
pred_lr = LR.predict(xv_test)
print("\n--- Model Değerlendirme Sonuçları ---")
print(f"Doğruluk Oranı (Accuracy): {accuracy_score(y_test, pred_lr)}")
print(classification_report(y_test, pred_lr))

print("6. Model ve Vektörizer kaydediliyor...")
# Daha sonra API'de kullanmak üzere modeli ve vektörizeri kaydedelim
with open('model.pkl', 'wb') as model_file:
    pickle.dump(LR, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorization, vec_file)

print("İşlem tamamlandı!")