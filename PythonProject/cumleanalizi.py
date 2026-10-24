import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences  # pad_sequences modern TF'de utils altındadır
import tensorflow as tf
dataset= pd.read_csv("hepsiburada.csv")
#target= değerlendirme puanları içeren liste
#data= inceleme metinlerini içeren liste
target=dataset['Rating'].values.tolist()
data= dataset['Review'].values.tolist()
cutoff=int(len(data)*0.8)
x_train,x_test=data[:cutoff],data[cutoff:]
y_train,y_test=target[:cutoff],target[cutoff:]
#data ve targetı yüzde 80 eğitim  yüzde 20 test olacak şekilde böldü cut offla
#print(x_train[500])
num_words=10000 # en çok kullanılan 10000 kelime
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data)
#print(tokenizer.word_index)
x_train_tokens=tokenizer.texts_to_sequences(x_train) #kelimeyi vektöre dönüştüren satır
#print(x_train_tokens[500])
x_test_tokens=tokenizer.texts_to_sequences(x_test)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens) #tüm veriyi tek bir liste haline getirip her bir cümlenin kaçar tokenden/kelimeden oluştuğunu tek tek hesaplıyor.
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
#VERİ SETİNDEKİ ORTALAMA TOKEN SAYISINI VE İKİ İLE ÇARPILMIŞ STANDART SAPMASINI (ORTALAMADAN SAPMASINI) BİR DEĞİŞKENE KAYDETMİŞ VE BUNU TAM SAYI HALİNE GETİRMİŞ.
np.sum(num_tokens < max_tokens) / len(num_tokens) #KAÇ TANE YORUMU KAPSADIĞINI BULMAK
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens) #AYNI TOKEN BOYUTUNA GETİRMEK İÇİN
#print(x_train_pad.shape) kaç tane yorum kaç token uzunluğunda onu gösteriyor
idx = tokenizer.word_index # ÖNCE KELİME VE NUMARALARIN BULUNDUĞU SÖZLÜĞÜ IDX DEĞİŞKENİNE ATIYORUM.
inverse_map = dict(zip(idx.values(), idx.keys())) # DAHA SONRA BU SÖZLÜKTEKİ VALUE VE KEY DEĞERLERİNİN YERİNİ DEĞİŞTİREREK YENİ BİR SÖZLÜK OLUŞTURUYORUM.
def tokens_to_string(tokens): # SAYILARDAN OLUŞAN BİR LİSTEYİ TEKRAR CÜMLEYE DÖNÜŞTÜREN FONKSİYON TANIMLIYORUM.
    words = [inverse_map[token] for token in tokens if token!=0]  # SAYILARI INVERSE_MAP SÖZLÜĞÜNDEKİ KELİMELERLE DEĞİŞTİRİYORUM.SAYI 0 İSE ATLAMALI ÇÜNKÜ SIFIRI TOKENLAR EŞİTLENSİN DİYE KOYDUK.
    text = ' '.join(words) # KELİMELERİ ARALARINA BOŞLUK KOYARAK TEK BİR CÜMLEDE BİRLEŞTİRİYORUM.
    return text # OLUŞTURDUĞUM CÜMLEYİ GERİ DÖNDÜRÜYORUM.
model=Sequential() # KATMANLI MODEL OLUŞTURMAYI SAĞLAR
embedding_size=50 # HER KELİMEYE KARŞILIK GELEN 50 UZUNLUĞUNDA VEKTÖR
model.add(Embedding(input_dim=num_words, output_dim=embedding_size, input_length=max_tokens,name='embedding_layer'))
#num_words MODELİN TANIYACAĞI BENZERSİZ KELİME SAYISI, embeding_size HER KELİMENİN DÖNÜŞECEĞİ VEKTÖR BOYUTU, MAX TOKEN UZUNLUĞU
# num_words MODELİN TANIYACAĞI BENZERSİZ KELİME SAYISI
# embedding_layer katman ismi
# HER BİR YORUM MATRİSLEŞİR GRU KATMANINA GİRMEYE HAZIR OLUR
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
# UNITS SİNİR AĞI SAYISI RETURN SEQUENCES İSE TÜMÜNÜ YAP VE DEVRET O YÜZDEN SON KATMANA VERMEDİK
model.add(Dense(1, activation='sigmoid'))
#GÖRSEL GÖSTERİMİ İÇİN DENSE KULLANDIK SİGMOİD DE 1 VE 0 DEĞERİ VERİYOR
optimizer = Adam(learning_rate=1e-3) #OPTİMİZER MODEL HATA YAPTIĞINDA YENİDEN DÜZENLER
# ADAM ALGORİTMASI MODELİ GÜNCELLERKEN GRADYAN BİLGİLER KULLANIR
# LR ÖĞRENME ORANIDIR 1E-3 İSE 0,001 ANLAMINA GELİR
model.compile(loss='binary_crossentropy', #LOSS MODELİN TAHMİNLERİ İLE GERÇEK ETİKETLER (TARGET) ARASINDAKİ HATAYI / SAPMAYI HESAPLAYAN MATEMATİKSEL FONKSİYONDUR.
              # BİNARY 0 VEYA 1 DEĞERLERİ İÇİNDİR
              optimizer=optimizer,#HATA FONKSİYONUNDAN (loss) ELDE EDİLEN SAPMAYA GÖRE, MODELDEKİ AĞIRLIKLARI (WEIGHTS) VE BİAS DEĞERLERİNİ NASIL GÜNCELLEYECEĞİNİ BELİRLEYEN ALGORİTMADIR.
              # GRADYAN (TÜREV) HESAPLAYARAK YAPAR
              metrics=['accuracy']) #NE KADAR BAŞARILI OLDUĞUNUN YÜZDESİ
#SANA VERDİĞİM YORUMLARI İKİLİ SINIFLANDIRMAYA (binary_crossentropy) GÖRE DEĞERLENDİR, YANLIŞLARINI SEÇTİĞİM OPTİMİZER İLE DÜZELT VE BANA BAŞARINI YÜZDE KAÇ (accuracy) OLARAK DÖNDÜR!
# Girdi boyutun (batch_size, max_tokens) şeklindedir
model.summary()
# Veriyi tf.Tensor formatına dönüştür:
x_train_pad = tf.convert_to_tensor(x_train_pad, dtype=tf.int32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
model.fit(x_train_pad,y_train, epochs=1, batch_size=256)
result = model.evaluate(x_test_pad, y_test)
#print(result[1]) test verisi doğruluk analizi
y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0] # TEST SETİNDEKİ İLK BİN YORUMU TAHMİN EDİP y_pred'e ATA.
cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])#TAHMİN OLASILIĞI 0.5 ' DEN BÜYÜKSE OLUMLU, DEĞİLSE OLUMSUZ YORUM.
cls_true = np.array(y_test[0:1000])
#İLK BİN ÖRNEĞİ OLUMLU VEYA OLUMSUZ OLDUĞUNU BELİRTEN BİR SINIFA KOYUYOR
incorrect = np.where(cls_pred != cls_true) #MODELİN TAHMİN ETTİĞİ VE GERÇEK CEVAP BİRBİRİNE EŞİT DEĞİLSE İNCORRECTE ATAR
incorrect = incorrect[0] #İLK ELEMANDAN ÇAĞIRIP SADECE SAYILARI ALIYOR
idx = incorrect[0]
