from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import string

# Uygulamayı başlat
app = FastAPI(title="Sahte Haber Tespit API")

# Kaydettiğimiz model ve vektörizeri yükle
print("Modeller yükleniyor...")
with open('model.pkl', 'rb') as model_file:
    LR = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorization = pickle.load(vec_file)

# Eğitirken kullandığımız metin temizleme fonksiyonunun aynısı
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

# Kullanıcıdan gelecek verinin yapısı
class NewsItem(BaseModel):
    text: str

@app.post("/predict")
def predict_news(news: NewsItem):
    # 1. Gelen metni temizle
    clean_text = wordopt(news.text)
    
    # 2. Metni modelin anlayacağı sayısal formata (vektöre) çevir
    vectorized_text = vectorization.transform([clean_text])
    
    # 3. Modeli kullanarak tahmin yap (0: Sahte, 1: Gerçek)
    prediction = LR.predict(vectorized_text)
    
    # 4. Sonucu okunabilir formata dönüştür
    result = "Gerçek Haber" if prediction[0] == 1 else "Sahte Haber"
    
    return {
        "status": "success",
        "prediction": result,
        "analyzed_text_snippet": news.text[:50] + "..." # Metnin başını loglama amaçlı dönüyoruz
    }