#GÖRSELDEN METİN ÇIKARAN MODEL EĞİTECEĞİZ. EĞİTİRKEN VGG16 MODELİNİN SON KATMANINI DEĞİŞTİRİCEZ. SIFIRDAN MODEL
# YAZMAK YERİNE SON KATMANI DEĞİŞTİRİP GÖRSEL VERİYİ KELİME ÇIKTISI OLARAK ALACAĞIZ
import time

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import coco
from coco import cache

#coco.maybe_download_and_extract() coco indirmek için 20 gb
"""COCO, özellikle bilgisayarlı görü ve görsel-dil modellerini eğitmek için kullanılan bir veri kümesidir. İçinde görseller ve bu görsellerdeki nesneler hakkında açıklayıcı bilgiler bulunur.
- Image Captioning: Bir görseli açıklayan cümle üretmek.
- Nesne tespiti: Görseldeki insan, araba, köpek gibi nesneleri bulmak.
- Görsel ve metin eşleştirme: Görsel içeriğiyle ilgili metinleri ilişkilendirmek."""
_, filenames, captions = coco.load_records(train=True)

# COCO VERİ KÜMESİNDEN EĞİTİM KAYITLARINI YÜKLE
# DÖNDÜRÜLEN İLK DEĞERİ KULLANMA
# DOSYA ADLARINI FİLENAMES DEĞİŞKENİNE ATA
# GÖRSEL AÇIKLAMALARINI CAPTIONS DEĞİŞKENİNE ATA
num_images = len(filenames)
#print(num_images) #KAÇ TANE RESİM VAR ONU YAZDIRDIK


def load_image(path, size=None): #RESMİN BOYUTU BELLİYSE YÜKLE AMA BELLİ DEĞİLSE LANCZOS İLE YENİDEN BOYUTLANDIR
    img = Image.open(path)

    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    return img
def show_image(idx):
    # GÖSTERİLECEK GÖRSELİN İNDEKSİNİ AL
    dir = coco.train_dir
    # EĞİTİM GÖRSELLERİNİN BULUNDUĞU KLASÖRÜ AL

    filename = filenames[idx]
    # İNDEKSE GÖRE GÖRSELİN DOSYA ADINI AL

    caption = captions[idx]
    # GÖRSELE AİT AÇIKLAMALARI AL

    path = os.path.join(dir, filename)
    # KLASÖR YOLU İLE DOSYA ADINI BİRLEŞTİREREK TAM YOLU OLUŞTUR

    for cap in caption:
        # GÖRSELE AİT HER AÇIKLAMAYI TEK TEK DÖNGÜYE AL
        print(cap)
        # AÇIKLAMAYI EKRANA YAZDIR

    #print(img = load_image(path))
    # DOSYA YOLUNU KULLANARAK GÖRSELİ YÜKLE

    #print(plt.imshow(img))
    # GÖRSELİ ÇİZİM ALANINDA GÖSTER

    #print(plt.show())
    # GÖRSELİ EKRANA YANSIT
image_model= VGG16 #VGG16 modelini yüklemek için 500 mb
#print(image_model.summary()) # modeli görüntülemek için
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
#son dense layerı dışarda bırakarak yeni model oluşturduk
img_size = K.int_shape(image_model.input)[1:3] # resmi hangi boyutlarda olmasını istediğimizi yazdırdık
print(img_size) # (224,224)
transfer_values_size = K.int_shape(transfer_layer.output)[1] #model çalıştırıldığında ne kadar vektör ürettiğini yazdırdık
print(transfer_values_size) #4096
#BÜTÜN VERİLERİ DEFALARCA ÖĞRETMEK YERİNE 1 DEFA GEÇİRİP ÇIKTIYI KAYDEDEM FONKSİYON

def print_progress(count, max_count):
    # TAMAMLANAN İŞLEM SAYISININ TOPLAM İŞLEM SAYISINA ORANINI HESAPLA
    pct_complete = count / max_count

    # TAMAMLANMA YÜZDESİNİ BİR ONDALIK BASAMAKLA GÖSTEREN MESAJI OLUŞTUR
    # \R İMLECİ SATIRIN BAŞINA GETİREREK MESAJIN AYNI SATIRDA GÜNCELLENMESİNİ SAĞLA
    msg = '\r- Progress: {0:.1%}'.format(pct_complete)

    # İLERLEME MESAJINI ÇIKTI AKIŞINA YAZ
    sys.stdout.write(msg)

    # ÇIKTIYI BEKLETMEDEN EKRANA AKTAR
    sys.stdout.flush()

def process_images(data_dir, filenames, batch_size=32, transfer_values=None):
    # İŞLENECEK TOPLAM GÖRSEL SAYISINI HESAPLA
    num_images = len(filenames)

    # BATCH BOYUTUNU, GÖRSEL BOYUTLARINI VE 3 RENK KANALINI BİRLEŞTİR
    shape = (batch_size,) + img_size + (3,)

    # GÖRSELLERİ TOPLU HALDE SAKLAMAK İÇİN SIFIRLARDAN OLUŞAN DİZİ OLUŞTUR
    image_batch = np.zeros(shape=shape, dtype=np.float16)

    # İŞLEME BAŞLANACAK İLK GÖRSELİN İNDEKSİNİ BELİRLE
    start_index = 0

    # TÜM GÖRSELLER İŞLENENE KADAR DÖNGÜYÜ SÜRDÜR
    while start_index < num_images:

        # İŞLEMİN İLERLEME YÜZDESİNİ EKRANA YAZDIR
        print_progress(count=start_index, max_count=num_images)

        # MEVCUT BATCH İÇİN BİTİŞ İNDEKSİNİ HESAPLA
        end_index = start_index + batch_size

        # BİTİŞ İNDEKSİ TOPLAM GÖRSEL SAYISINI AŞARSA SINIRLA
        if end_index > num_images:
            end_index = num_images

        # MEVCUT BATCH İÇİNDE KAÇ GÖRSEL OLDUĞUNU HESAPLA
        current_batch_size = end_index - start_index

        # MEVCUT BATCH İÇİNDEKİ GÖRSELLERİ TEK TEK İŞLE
        for i, filename in enumerate(filenames[start_index:end_index]):

            # VERİ KLASÖRÜ İLE DOSYA ADINI BİRLEŞTİREREK DOSYA YOLUNU OLUŞTUR
            path = os.path.join(data_dir, filename)

            # GÖRSELİ YÜKLE VE HEDEF BOYUTA GETİR
            img = load_image(path, size=img_size)

            # YÜKLENEN GÖRSELİ BATCH DİZİSİNDEKİ İLGİLİ KONUMA YERLEŞTİR
            image_batch[i] = img

        # BATCH İÇİNDEKİ GÖRSELLERDEN MODELİN ÖZELLİK VEKTÖRLERİNİ ÇIKAR
        transfer_values_batch = image_model_transfer.predict(
            image_batch[0:current_batch_size]
        )

        # ELDE EDİLEN ÖZELLİK VEKTÖRLERİNİ ANA DİZİDEKİ İLGİLİ KONUMA KAYDET
        transfer_values[start_index:end_index] = transfer_values_batch[0:current_batch_size]

        # SONRAKİ BATCH'İN BAŞLANGIÇ İNDEKSİNİ GÜNCELLE
        start_index = end_index

    # İLERLEME BİLGİSİNDEN SONRA ALT SATIRA GEÇ
    print()

    # TÜM GÖRSELLER İÇİN HESAPLANAN ÖZELLİK VEKTÖRLERİNİ GERİ DÖNDÜR
    return transfer_values
def process_train_images():
    # EĞİTİM SETİNDEKİ TOPLAM GÖRSEL SAYISINI EKRANA YAZDIR
    print('Eğitim setindeki {0} resim işleniyor...'.format(len(filenames)))

    # ÖZELLİK VEKTÖRLERİNİN ÖNBELLEK DOSYASININ YOLUNU OLUŞTUR
    cache_path = os.path.join(coco.data_dir, 'transfer_values_train.pkl')

    # GÖRSELLERİ İŞLE VE SONUÇLARI ÖNBELLEKTE SAKLA
    # ÖNBELLEKTE VERİ VARSA YENİDEN İŞLEMEK YERİNE MEVCUT VERİYİ KULLAN
    transfer_values = cache(
        cache_path=cache_path,
        fn=process_images,
        data_dir=coco.train_dir,
        filenames=filenames
    )

    # EĞİTİM GÖRSELLERİNDEN ELDE EDİLEN ÖZELLİK VEKTÖRLERİNİ GERİ DÖNDÜR
    return transfer_values


transfer_values = process_train_images()
# EĞİTİM GÖRSELLERİNİ İŞLE VE ÖZELLİK VEKTÖRLERİNİ ELDE ET

print('Shape:', transfer_values.shape)
# ÖZELLİK VEKTÖRLERİ DİZİSİNİN BOYUTLARINI EKRANA YAZDIR
mark_start = 'ssss ' #BAŞLANGIÇ VE BİTİŞ TOKENLARI
mark_end = ' eeee'
# COCODA 5 TANE AÇIKLAMA VE GÖRSEL VERİYORDU. AÇIKLAMAYI TOKENLEŞTIRMEK İÇİN BAŞLANGIÇ VE BİTİŞ TOKENLARINI EKLEYECEĞİZ
def mark_captions(captions_listlist):
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]
    return captions_marked
captions_marked = mark_captions(captions) #DEĞİŞKENE EKLEDİK
#LİSTE İÇERİSİNDEKİ LİSTELERİ TEK BİR CAPTIONA YAZMAK İÇİN DÖNGÜ
def flatten(captions_listlist):
    # İÇ İÇE LİSTELERDEKİ TÜM AÇIKLAMALARI TEK BİR LİSTEDE TOPLA
    captions_list = [caption
                     # HER BİR AÇIKLAMA LİSTESİNİ DÖNGÜYE AL
                     for captions_list in captions_listlist
                     # HER LİSTENİN İÇİNDEKİ AÇIKLAMALARI TEK TEK AL
                     for caption in captions_list]

    # DÜZLEŞTİRİLMİŞ AÇIKLAMA LİSTESİNİ GERİ DÖNDÜR
    return captions_list #COCO veri kümesindeki görsellere ait açıklamaları tek bir listede toplamak için kullanılır.
captions_flat = flatten(captions_marked)
num_words=10000 #CAPTIONLARDAKİ EN YAYGIN 10000 KELİME
class TokenizerWrap(Tokenizer):  # CP CAPTIONS_TO_TOKENS'İ KOPYALAMA, YAZILACAK

    def __init__(self, texts, num_words=None):
        # TOKENIZER SINIFINI BAŞLAT VE KELİME SAYISI SINIRINI BELİRLE
        Tokenizer.__init__(self, num_words=num_words)

        # METİNLERİ ANALİZ EDEREK KELİME SÖZLÜĞÜNÜ OLUŞTUR
        self.fit_on_texts(texts)

        # KELİME İNDEKSLERİNİ KELİMELERE EŞLEYEN SÖZLÜK OLUŞTUR
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        # TOKEN 0 İSE BOŞLUK, DEĞİLSE TOKENIN KARŞILIĞI OLAN KELİMEYİ AL
        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        # SIFIR OLMAYAN TOKENLARI KELİMELERE DÖNÜŞTÜR
        words = [self.index_to_word[token] for token in tokens if token != 0]

        # KELİMELERİ BOŞLUKLARLA BİRLEŞTİREREK CÜMLE OLUŞTUR
        text = " ".join(words)

        # OLUŞTURULAN CÜMLEYİ GERİ DÖNDÜR
        return text

    def captions_to_tokens(self, captions_listlist):
        # HER GÖRSELİN AÇIKLAMALARINI TOKEN NUMARALARINA DÖNÜŞTÜR
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        return tokens
#BU SINIF KELİMELERİ İNDEKSLERE DÖNÜŞTÜRÜYOR VE ÜRETİLEN TOKENLARI KELİMELERE DÖNÜŞTÜRÜYOR
# METİN AÇIKLAMALARINDAN TOKENIZER OLUŞTUR VE KELİME SAYISINI SINIRLA
tokenizer = TokenizerWrap(texts=captions_flat, num_words=num_words)

# İŞARETLENMİŞ EĞİTİM AÇIKLAMALARINI TOKEN NUMARALARINA DÖNÜŞTÜR
tokens_train = tokenizer.captions_to_tokens(captions_marked)

# BAŞLANGIÇ VE BİTİŞ İŞARETLERİNİN TOKEN NUMARALARINI AL
token_start = tokenizer.word_index[mark_start.strip()]
token_end = tokenizer.word_index[mark_end.strip()]

#HAFIZA SORUNU YAŞAMAMAK İÇİN HER BİR RESİME RASTGELE BİR AÇIKLAMA SEÇİYOR
def get_random_caption_tokens(idx):
    # VERİLEN GÖRSEL İNDEKSLERİ İÇİN RASTGELE AÇIKLAMALAR SEÇ
    result = []

    # HER GÖRSELİN İNDEKSİNİ TEK TEK İŞLE
    for i in idx:
        # GÖRSELE AİT AÇIKLAMALAR ARASINDAN RASTGELE BİRİNİ SEÇ
        j = np.random.choice(len(tokens_train[i]))

        # SEÇİLEN AÇIKLAMANIN TOKENLARINI AL
        tokens = tokens_train[i][j]

        # SEÇİLEN TOKEN LİSTESİNİ SONUÇ LİSTESİNE EKLE
        result.append(tokens)

    # RASTGELE SEÇİLEN AÇIKLAMALARIN TOKENLARINI GERİ DÖNDÜR
    return result


def batch_generator(batch_size):
    # EĞİTİM İÇİN SÜREKLİ OLARAK YENİ BATCH'LER ÜRET
    while True:
        # EĞİTİM GÖRSELLERİ ARASINDAN RASTGELE İNDEKSLER SEÇ
        idx = np.random.randint(num_images, size=batch_size)

        # SEÇİLEN GÖRSELLERİN ÖZELLİK VEKTÖRLERİNİ AL #RESSİMLERİN TRANSFER DEĞERLERİNİ KULLANIYORUZ
        t_values = transfer_values[idx]

        # SEÇİLEN GÖRSELLER İÇİN RASTGELE AÇIKLAMALARIN TOKENLARINI AL
        tokens = get_random_caption_tokens(idx)

        # HER AÇIKLAMADAKİ TOKEN SAYISINI HESAPLA
        num_tokens = [len(t) for t in tokens]

        # BATCH İÇİNDEKİ EN UZUN AÇIKLAMANIN TOKEN SAYISINI BUL
        #print(max_tokens = np.max(num_tokens))

        # AÇIKLAMALARI AYNI UZUNLUĞA GETİR VE EKSİK KISIMLARI SIFIRLA DOLDUR
        tokens_padded = pad_sequences(
            tokens,
            maxlen=max_tokens,
            padding='post',
            truncating='post'
        )

        # DECODER GİRDİSİNİ OLUŞTURARAK SON TOKENI ÇIKAR
        decoder_input_data = tokens_padded[:, 0:-1]

        # HEDEF ÇIKTIYI OLUŞTURARAK İLK TOKENI ÇIKAR
        decoder_output_data = tokens_padded[:, 1:]

        # MODELİN GİRDİLERİNİ GÖRSEL ÖZELLİKLERİ VE DECODER TOKENLARI OLARAK HAZIRLA
        x_data = {
            'decoder_input': decoder_input_data,
            'transfer_values_input': t_values
        }

        # MODELİN ÖĞRENMESİ İÇİN HEDEF TOKENLARI HAZIRLA
        y_data = {'decoder_output': decoder_output_data}

        # GİRDİ VE HEDEF VERİLERİNİ EĞİTİM İÇİN GERİ DÖNDÜR
        yield (x_data, y_data)


# HER EĞİTİM BATCH'İNDE 256 ÖRNEK KULLAN
batch_size = 256

# BATCH ÜRETİCİSİNİ OLUŞTUR
generator = batch_generator(batch_size)

# ÜRETİCİDEN İLK BATCH'İ AL
batch = next(generator)

# BATCH'İN GİRDİ VERİLERİNİ AL
batch_x = batch[0]

# BATCH'İN HEDEF ÇIKTI VERİLERİNİ AL
batch_y = batch[1]
num_captions = [len(caption) for caption in captions] #her listede kaç captıon bulunduğunu listeye ekliyoruz
total_num_captions = np.sum(num_captions) #kaç tane captıon bulunduğunu hesapladık
steps_per_epoch = int(total_num_captions / batch_size) #bir epoch için gerekli iterasyon sayısı
state_size=256 #GRU'nun her zaman adımında tuttuğu hafıza vektörünün boyutu
embedding_size=100 # kullanılacak kelime vektörlerinin uzunluğu 100
# GÖRSELDEN GELEN ÖZELLİK VEKTÖRÜNÜ MODELİN GİRDİSİ OLARAK TANIMLA
transfer_values_input = Input(
    shape=(transfer_values_size,),
    name='transfer_values_input'
)

# GÖRSEL ÖZELLİKLERİNİ GRU'NUN STATE BOYUTUNA DÖNÜŞTÜR
decoder_transfer_map = Dense(
    state_size,
    activation='tanh',
    name='decoder_transfer_map'
)

# DECODER'A VERİLECEK TOKEN GİRDİSİNİ TANIMLA
decoder_input = Input(
    shape=(None,),
    name='decoder_input'
)

# GLOVE KELİME VEKTÖRLERİNİ SAKLAYACAK SÖZLÜĞÜ OLUŞTUR
word2vec = {}

# GLOVE DOSYASINI OKU
with open('glove.6B.100d.txt', encoding='UTF-8') as f:

    # DOSYADAKİ HER SATIRI TEK TEK İŞLE
    for line in f:

        # SATIRI KELİME VE VEKTÖR DEĞERLERİNE AYIR
        values = line.split()

        # SATIRDAKİ İLK DEĞERİ KELİME OLARAK AL
        word = values[0]

        # KELİMEDEN SONRAKİ 100 SAYIYI VEKTÖRE DÖNÜŞTÜR
        vec = np.asarray(values[1:], dtype='float32')

        # KELİMEYİ KENDİ GLOVE VEKTÖRÜYLE SÖZLÜĞE KAYDET
        word2vec[word] = vec

# TOKENLAR İÇİN RASTGELE BAŞLANGIÇ EMBEDDING MATRİSİ OLUŞTUR
embedding_matrix = np.random.uniform(
    -1, 1,
    (num_words, embedding_size)
)

# TOKENIZER'DAKİ HER KELİMEYİ TEK TEK İŞLE
for word, i in tokenizer.word_index.items():

    # KELİME TOKEN SINIRI İÇİNDEYSE İŞLE
    if i < num_words:

        # KELİMENİN GLOVE'DAKİ VEKTÖRÜNÜ BUL
        embedding_vector = word2vec.get(word)

        # KELİME GLOVE'DA BULUNUYORSA VEKTÖRÜ KULLAN
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# KELİMELERİ SAYISAL VEKTÖRLERE DÖNÜŞTÜREN EMBEDDING KATMANINI OLUŞTUR
decoder_embedding = Embedding(
    input_dim=num_words,
    output_dim=embedding_size,
    weights=[embedding_matrix],
    trainable=False,
    name='decoder_embedding'
)

# İLK GRU KATMANINI OLUŞTUR
decoder_gru1 = GRU(
    state_size,
    name='decoder_gru1',
    return_sequences=True
)

# İKİNCİ GRU KATMANINI OLUŞTUR
decoder_gru2 = GRU(
    state_size,
    name='decoder_gru2',
    return_sequences=True
)

# ÜÇÜNCÜ GRU KATMANINI OLUŞTUR
decoder_gru3 = GRU(
    state_size,
    name='decoder_gru3',
    return_sequences=True
)

# GRU ÇIKTILARINI KELİME TOKENLARINA DÖNÜŞTÜREN ÇIKIŞ KATMANINI OLUŞTUR
decoder_dense = Dense(
    num_words,
    activation='linear',
    name='decoder_output'
)


def connect_decoder(transfer_values):
    # GÖRSEL ÖZELLİKLERİNİ DECODER'IN BAŞLANGIÇ STATE'İNE DÖNÜŞTÜR
    initial_state = decoder_transfer_map(transfer_values)

    # DECODER GİRDİSİNİ BAŞLANGIÇ OLARAK AL
    net = decoder_input

    # TOKENLARI GLOVE EMBEDDING VEKTÖRLERİNE DÖNÜŞTÜR
    net = decoder_embedding(net)

    # EMBEDDINGLERİ İLK GRU KATMANINDAN GEÇİR
    net = decoder_gru1(net, initial_state=initial_state)

    # İLK GRU ÇIKTISINI İKİNCİ GRU KATMANINDAN GEÇİR
    net = decoder_gru2(net, initial_state=initial_state)

    # İKİNCİ GRU ÇIKTISINI ÜÇÜNCÜ GRU KATMANINDAN GEÇİR
    net = decoder_gru3(net, initial_state=initial_state)

    # HER TOKEN İÇİN KELİME TAHMİNİ ÜRET
    decoder_output = decoder_dense(net)

    # DECODER ÇIKTISINI GERİ DÖNDÜR
    return decoder_output


# TRANSFER VALUES GİRDİSİNİ DECODER'A BAĞLA
decoder_output = connect_decoder(
    transfer_values=transfer_values_input
)

# TRANSFER VALUES VE DECODER INPUT'U MODELİN GİRDİLERİ OLARAK TANIMLA
# DECODER OUTPUT'U MODELİN ÇIKIŞI OLARAK BELİRLE
decoder_model = Model(
    inputs=[transfer_values_input, decoder_input],
    outputs=[decoder_output]
)
def sparse_cross_entropy(y_true, y_pred):
    # GERÇEK TOKENLAR İLE MODELİN TAHMİN ETTİĞİ TOKENLAR ARASINDAKİ HATAYI HESAPLA
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=y_pred
    )

    # HESAPLANAN TÜM HATALARIN ORTALAMASINI AL
    loss_mean = tf.reduce_mean(loss)

    # ORTALAMA HATA DEĞERİNİ GERİ DÖNDÜR
    return loss_mean
# RMSPROP OPTIMIZER'INI 0.001 ÖĞRENME ORANIYLA OLUŞTUR
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

# DECODER MODELİNİ OPTIMIZER VE KAYIP FONKSİYONUYLA DERLE
# GERÇEK HEDEF TOKEN'LARI EĞİTİM SIRASINDA FIT METODUYLA VERİLECEK
decoder_model.compile(
    optimizer=optimizer,
    loss=sparse_cross_entropy
)

# MODEL AĞIRLIKLARINI KAYDETMEK İÇİN DOSYA ADINI BELİRLE
# GÜNCEL KERAS, YALNIZCA AĞIRLIK KAYDEDİLİRKEN BU UZANTIYI BEKLER
path_checkpoint = 'checkpoint.weights.h5'

# HER EPOCH SONUNDA MODEL AĞIRLIKLARINI KAYDET
checkpoint = ModelCheckpoint(
    filepath=path_checkpoint,
    save_weights_only=True,
    save_best_only=False
)

# DAHA ÖNCE KAYDEDİLMİŞ MODEL AĞIRLIKLARINI YÜKLEMEYİ DENE
try:
    decoder_model.load_weights(path_checkpoint)

# DOSYA BULUNAMAZSA VEYA AĞIRLIKLAR UYUŞMAZSA HATAYI GÖSTER
except (OSError, ValueError) as error:
    print("CHECKPOINT YÜKLENEMEDİ. EĞİTİM BAŞLANGIÇ AĞIRLIKLARIYLA DEVAM EDECEK.")
    print(error)

# GENERATOR'IN (GİRDİLER, HEDEFLER) ŞEKLİNDE VERİ DÖNDÜRDÜĞÜNÜ VARSAYARAK EĞİT
# GİRDİLER MODELİN GİRDİ İSİMLERİYLE UYUŞMALI, HEDEFLER GERÇEK TOKEN KİMLİKLERİ OLMALI
decoder_model.fit(
    generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    callbacks=[checkpoint]
)


# VERİLEN GÖRSEL İÇİN AÇIKLAMA ÜRETEN FONKSİYONU TANIMLA
def generate_caption(image_path, max_tokens=30):

    # GÖRSELİ YÜKLE VE MODELİN BEKLEDİĞİ BOYUTA GETİR
    image = load_image(image_path, size=img_size)

    # MODELE BATCH BOYUTUNU EKLE
    image_batch = np.expand_dims(image, axis=0)

    # GÖRSELİN ÖZELLİKLERİNİ ÖNCEDEN EĞİTİLMİŞ GÖRSEL MODELİNDEN ÇIKAR
    transfer_values = image_model_transfer.predict(
        image_batch,
        verbose=0
    )

    # DECODER GİRDİSİ İÇİN BOŞ TOKEN DİZİSİ OLUŞTUR
    decoder_input_data = np.zeros(
        shape=(1, max_tokens),
        dtype=np.int32
    )

    # AÇIKLAMAYI BAŞLANGIÇ TOKEN'IYLA ÜRETMEYE BAŞLA
    token_int = token_start
    output_text = ''
    count_tokens = 0

    # BİTİŞ TOKEN'I GELENE VEYA TOKEN SINIRINA ULAŞILANA KADAR DEVAM ET
    while token_int != token_end and count_tokens < max_tokens:

        # MEVCUT TOKEN'I DECODER GİRDİSİNİN İLGİLİ KONUMUNA YAZ
        decoder_input_data[0, count_tokens] = token_int

        # GÖRSEL ÖZELLİKLERİNİ VE TOKEN DİZİSİNİ MODEL GİRDİSİ OLARAK HAZIRLA
        x_data = {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        # MODELİN KELİME TAHMİNLERİNİ HESAPLA
        decoder_output = decoder_model.predict(
            x_data,
            verbose=0
        )

        # MEVCUT KONUM İÇİN TAHMİN DEĞERLERİNİ AL
        token_onehot = decoder_output[0, count_tokens, :]

        # EN YÜKSEK TAHMİN DEĞERİNE SAHİP TOKEN'I SEÇ
        token_int = int(np.argmax(token_onehot))

        # BİTİŞ TOKEN'INI AÇIKLAMA METNİNE EKLEMEDEN DÖNGÜYÜ SONLANDIR
        if token_int == token_end:
            break

        # TOKEN KİMLİĞİNİ KELİMEYE DÖNÜŞTÜR
        sampled_word = tokenizer.token_to_word(token_int)

        # TAHMİN EDİLEN KELİMEYİ AÇIKLAMAYA EKLE
        output_text += " " + str(sampled_word)

        # ÜRETİLEN TOKEN SAYACINI ARTIR
        count_tokens += 1

    # GÖRSELİ EKRANDA GÖSTER
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # MODELİN ÜRETTİĞİ AÇIKLAMAYI YAZDIR
    print('PREDICTED CAPTION:')
    print(output_text.strip())
    print()


# COCO VERİ KÜMESİNDEN BİR GÖRSEL SEÇİP AÇIKLAMASINI ÜRET
def generate_caption_coco(idx):

    # COCO EĞİTİM GÖRSELLERİNİN BULUNDUĞU DİZİNİ AL
    data_dir = coco.train_dir

    # SEÇİLEN İNDEKSE AİT DOSYA ADINI VE GERÇEK AÇIKLAMALARI AL
    filename = filenames[idx]
    caption = captions[idx]

    # GÖRSELİN TAM DOSYA YOLUNU OLUŞTUR
    path = os.path.join(data_dir, filename)

    # MODELİN SEÇİLEN GÖRSEL İÇİN AÇIKLAMA ÜRETMESİNİ SAĞLA
    generate_caption(image_path=path)

    # VERİ KÜMESİNDEKİ GERÇEK AÇIKLAMALARI GÖSTER
    print('TRUE CAPTIONS:')
    for cap in caption:
        print(cap)
