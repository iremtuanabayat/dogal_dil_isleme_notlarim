import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Embedding, GRU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

mark_start = 'ssss '
#'ssss ' (Start of Sequence / Başlangıç Etiketi):
# Decoder (çözücü) katmanına üretimi başlatması için verilen tetikleyici sinyaldir.
# Model henüz hiç kelime üretmemişken "çeviriye başla" komutunu temsil eder.
mark_end = ' eeee'
#' eeee' (End of Sequence / Bitiş Etiketi):
#Cümlenin veya dizinin tamamlandığını belirten bitiş sınırıdır.
# Decoder bu etiketi ürettiğinde cümlenin bittiğini anlar ve tahmini durdurur.
data_src = [] #ingilizce için boş liste
data_dest = [] #hedef çeviri türkçe için boş liste
for line in open('tur.txt', encoding='UTF-8'):
    en_text, tr_text = line.rstrip().split('\t')

    tr_text = mark_start + tr_text + mark_end

    data_src.append(en_text)
    data_dest.append(tr_text)
# tur.txt dosyasının satırlarını teker teker gezerek tab tuşuyla bir ayrılan blokları listeye ekler.
class TokenizerWrap(Tokenizer):  # TOKENIZER SINIFINDAN MİRAS ALAN YENİ SINIF

    def __init__(self, texts, padding, reverse=False, num_words=None):
        # SINIF OLUŞTURULDUĞUNDA ÇALIŞAN BAŞLANGIÇ FONKSİYONU

        Tokenizer.__init__(self, num_words=num_words)
        # ÜST SINIF OLAN TOKENIZER'I BAŞLAT

        self.fit_on_texts(texts)
        # VERİLEN METİNLERDEKİ KELİMELERİ ÖĞREN VE KELİME-SAYI SÖZLÜĞÜ OLUŞTUR

        self.index_to_word = dict(
            zip(self.word_index.values(), self.word_index.keys())
        )
        # KELİME-SAYI SÖZLÜĞÜNÜ SAYI-KELİME SÖZLÜĞÜNE TERS ÇEVİR

        self.tokens = self.texts_to_sequences(texts)
        # TÜM METİNLERİ TOKEN ID'LERİNE ÇEVİR

        if reverse:
            # REVERSE TRUE İSE TOKEN SIRASINI TERS ÇEVİR

            self.tokens = [list(reversed(x)) for x in self.tokens]
            # HER CÜMLENİN TOKEN SIRASINI TERSİNE ÇEVİR

            truncating = 'pre'
            # FAZLA UZUN CÜMLELERİN BAŞINDAN KES

        else:
            truncating = 'post'
            # FAZLA UZUN CÜMLELERİN SONUNDAN KES

        self.num_tokens = [len(x) for x in self.tokens]
        # HER CÜMLENİN KAÇ TOKEN'DAN OLUŞTUĞUNU HESAPLA

        self.max_tokens = np.mean(self.num_tokens) + 2 * np.std(self.num_tokens)
        # ORTALAMA + 2 STANDART SAPMA İLE MAKSİMUM TOKEN SAYISINI HESAPLA
        tokenizer_src = TokenizerWrap(  # KAYNAK METİNLER İÇİN TOKENIZER NESNESİ OLUŞTUR
            texts=data_src,  # İŞLENECEK KAYNAK METİNLERİ BELİRLE
            padding='pre',  # EKSİK TOKENLARI DİZİNİN BAŞINA EKLE
            reverse=True,  # TOKEN SIRASINI TERSİNE ÇEVİR
            num_words=None  # KELİME SAYISINA EK BİR SINIR KOYMA
        )
        self.max_tokens = int(self.max_tokens)
        # HESAPLANAN DEĞERİ TAM SAYIYA ÇEVİR

        self.tokens_padded = pad_sequences(
            self.tokens,
            maxlen=self.max_tokens,
            padding=padding,
            truncating=truncating
        )
        # TÜM CÜMLELERİ AYNI UZUNLUĞA GETİRMEK İÇİN PADDING UYGULA


    def token_to_word(self, token):
        # TEK BİR TOKEN'I KELİMEYE ÇEVİR

        word = ' ' if token == 0 else self.index_to_word[token]
        # TOKEN 0 İSE BOŞLUK, DEĞİLSE TOKEN'IN KARŞILIĞI OLAN KELİMEYİ AL

        return word
        # KELİMEYİ GERİ DÖNDÜR


    def tokens_to_string(self, tokens):
        # TOKEN LİSTESİNİ TEKRAR METNE ÇEVİR

        words = [self.index_to_word[token] for token in tokens if token != 0]
        # 0 OLMAYAN TOKEN'LARI KELİMELERE ÇEVİR

        text = ' '.join(words)
        # KELİMELERİ BOŞLUKLARLA BİRLEŞTİREREK CÜMLE OLUŞTUR

        return text
        # OLUŞTURULAN METNİ GERİ DÖNDÜR


    def text_to_tokens(self, text, padding, reverse=False):
        # YENİ BİR METNİ MODELİN KULLANABİLECEĞİ TOKEN FORMATINA ÇEVİR

        tokens = self.texts_to_sequences([text])
        # VERİLEN METNİ TOKEN ID'LERİNE ÇEVİR

        tokens = np.array(tokens)
        # TOKEN LİSTESİNİ NUMPY ARRAY'E ÇEVİR

        if reverse:
            # REVERSE TRUE İSE TOKEN SIRASINI TERS ÇEVİR

            tokens = np.flip(tokens, axis=1)
            # TOKEN'LARI 1. EKSENDE TERS ÇEVİR

            truncating = 'pre'
            # FAZLA TOKEN VARSA BAŞTAN KES

        else:
            truncating = 'post'
            # FAZLA TOKEN VARSA SONDAN KES

        tokens = pad_sequences(
            tokens,
            maxlen=self.max_tokens,
            padding=padding,
            truncating=truncating
        )
        # YENİ METNİ DE MODELİN BEKLEDİĞİ UZUNLUĞA GETİR

        return tokens
        # HAZIRLANAN TOKEN'LARI GERİ DÖNDÜR

tokenizer_src = TokenizerWrap(  # KAYNAK METİNLER İÇİN TOKENIZER NESNESİ OLUŞTUR
        texts=data_src,  # İŞLENECEK KAYNAK METİNLERİ BELİRLE
        padding='pre',  # EKSİK TOKENLARI DİZİNİN BAŞINA EKLE
        reverse=True,  # TOKEN SIRASINI TERSİNE ÇEVİR
        num_words=None  # KELİME SAYISINA EK BİR SINIR KOYMA
    )
tokenizer_dest = TokenizerWrap(  # KAYNAK METİNLER İÇİN TOKENIZER NESNESİ OLUŞTUR
    texts=data_dest,       # İŞLENECEK KAYNAK METİNLERİ BELİRLE
    padding='post',        # EKSİK TOKENLARI DİZİNİN BAŞINA EKLE
    reverse=False,         # TOKEN SIRASINI TERSİNE ÇEVİR
    num_words=None        # KELİME SAYISINA EK BİR SINIR KOYMA
)
tokens_src = tokenizer_src.tokens_padded # TOKENİZE EDİP DEĞİŞKENLERE ATADIK
tokens_dest = tokenizer_dest.tokens_padded
# Başlangıç işaretinin başındaki ve sonundaki boşlukları temizler ve numarasını bulur.
token_start = tokenizer_dest.word_index[mark_start.strip()]

# Bitiş işaretinin başındaki ve sonundaki boşlukları temizler ve numarasını bulur.
token_end = tokenizer_dest.word_index[mark_end.strip()]

# Kaynak metinlerin tokenlara dönüştürülmüş hâlini encoder giriş verisi olarak atar.
encoder_input_data = tokens_src

# Hedef metinlerin son tokenını çıkararak decoder giriş verisini oluşturur.
decoder_input_data = tokens_dest[:, :-1]

# Hedef metinlerin ilk tokenını çıkararak decoder çıkış verisini oluşturur.
decoder_output_data = tokens_dest[:, 1:]

# Kaynak dildeki sözlükte bulunan toplam kelime sayısını hesaplar.
num_encoder_words = len(tokenizer_src.word_index)

# Hedef dildeki sözlükte bulunan toplam kelime sayısını hesaplar.
num_decoder_words = len(tokenizer_dest.word_index)
embedding_size=100 #KELİME VEKTÖRLERİNİN UZUNLUĞU
word2vec={}
with open('glove.6B.100d.txt', encoding='UTF-8') as f: # GLOVE DOSYASINI AÇ f OLARAK YÜKLE
    for line in f: #DÖNGÜ HER DÖNDÜĞÜNDE HER SATIRI f 'e ATA.
        values = line.split()
        word = values[0]
        vec=np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec #KELİMELERİ VE VEKTÖLERİNİ BİR SÖZLÜKTE TOPLADIK
embedding_matrix = np.random.unıform(-1,1,num_encoder_words,embedding_size)
for word, i in tokenizer_src.word_index.items(): #DÖNGÜ HER DÖNDÜĞÜNDE KELİME WORDE SAYİ İSE VECTÖRE ATANIYOR
    if i < num_encoder_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
encoder_input = Input(shape=(None,), name='encoder_input')
# Encoder giriş katmanını oluşturur.
# shape=(None,) giriş dizisinin uzunluğunun önceden sabitlenmediğini belirtir.
# name='encoder_input' katmana 'encoder_input' adını verir.
encoder_embedding = Embedding(input_dim=num_encoder_words,
                              output_dim=embedding_size,
                              weights=[embedding_matrix],
                              trainable=True,
                              name='encoder_embedding')
state_size = 256
encoder_gru1 = GRU(state_size, name='encoder_gru1', return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2', return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3', return_sequences=False)


# Encoder katmanlarını birbirine bağlayan fonksiyonu tanımlar.
def connect_encoder():

    # Encoder giriş verisini ağın başlangıç noktası olarak belirler.
    net = encoder_input

    # Giriş verisini embedding katmanından geçirerek sayısal vektörlere dönüştürür.
    net = encoder_embedding(net)

    # Veriyi birinci GRU katmanından geçirerek sıralı bilgileri işler.
    net = encoder_gru1(net)

    # Veriyi ikinci GRU katmanından geçirerek sıralı bilgileri işlemeye devam eder.
    net = encoder_gru2(net)

    # Veriyi üçüncü GRU katmanından geçirerek işlemi tamamlar.
    net = encoder_gru3(net)

    # Son GRU katmanının çıktısını encoder_output değişkenine atar.
    encoder_output = net

    # Encoder'ın ürettiği çıktıyı fonksiyonun dışına döndürür.
    return encoder_output


# Encoder fonksiyonunu çalıştırarak encoder çıktısını elde eder.
encoder_output = connect_encoder()


# Decoder'ın başlangıç gizli durumunu alacak giriş katmanını oluşturur.
# shape=(state_size,) gizli durum vektörünün boyutunu belirtir.
# name='decoder_initial_state' katmana bir isim verir.
decoder_initial_state = Input(shape=(state_size,), name='decoder_initial_state')


# Decoder'ın hedef dizilerden gelen tokenları alacağı giriş katmanını oluşturur.
# shape=(None,) dizi uzunluğunun sabit olmadığını belirtir.
decoder_input = Input(shape=(None,), name='decoder_input')


# Decoder tokenlarını embedding vektörlerine dönüştürecek katmanı oluşturur.
# input_dim hedef sözlükteki kelime sayısını belirtir.
# output_dim her tokenın temsil edileceği vektörün boyutunu belirtir.
# name katmanın adını belirler.
decoder_embedding = Embedding(
    input_dim=num_decoder_words,
    output_dim=embedding_size,
    name='decoder_embedding'
)


# Decoder'ın birinci GRU katmanını oluşturur.
# state_size gizli durum vektörünün boyutunu belirler.
# return_sequences=True tüm zaman adımları için çıktı üretilmesini sağlar.
decoder_gru1 = GRU(
    state_size,
    name='decoder_gru1',
    return_sequences=True
)


# Decoder'ın ikinci GRU katmanını oluşturur.
# Önceki katmandan gelen sıralı çıktıları işler.
# return_sequences=True tüm zaman adımlarının çıktısını korur.
decoder_gru2 = GRU(
    state_size,
    name='decoder_gru2',
    return_sequences=True
)


# Decoder'ın üçüncü GRU katmanını oluşturur.
# Önceki katmandan gelen sıralı çıktıları işler.
# return_sequences=True her zaman adımı için çıktı üretir.
decoder_gru3 = GRU(
    state_size,
    name='decoder_gru3',
    return_sequences=True
)


# Decoder'ın çıktı katmanını oluşturur.
# Dense katmanı her zaman adımında hedef sözlükteki her kelime için bir değer üretir.
# num_decoder_words çıktı değerlerinin sayısını, yani hedef sözlük boyutunu belirtir.
# activation='linear' doğrusal aktivasyon fonksiyonunu kullanır.
# name='decoder_output' katmanın adını belirler.
decoder_dense = Dense(
    num_decoder_words,
    activation='linear',
    name='decoder_output'
)


# DECODER KATMANLARINI BİRBİRİNE BAĞLAYAN FONKSİYONU TANIMLAR.
def connect_decoder(initial_state):

    # DECODER GİRİŞ VERİSİNİ AĞIN BAŞLANGIÇ NOKTASI OLARAK BELİRLER.
    net = decoder_input

    # GİRİŞ TOKENLARINI SAYISAL VEKTÖRLERE DÖNÜŞTÜRÜR.
    net = decoder_embedding(net)

    # VERİYİ BİRİNCİ GRU KATMANINDAN GEÇİRİR VE BAŞLANGIÇ GİZLİ DURUMUNU ATAR.
    net = decoder_gru1(net, initial_state=initial_state)

    # VERİYİ İKİNCİ GRU KATMANINDAN GEÇİRİR VE BAŞLANGIÇ GİZLİ DURUMUNU ATAR.
    net = decoder_gru2(net, initial_state=initial_state)

    # VERİYİ ÜÇÜNCÜ GRU KATMANINDAN GEÇİRİR VE BAŞLANGIÇ GİZLİ DURUMUNU ATAR.
    net = decoder_gru3(net, initial_state=initial_state)

    # DECODER ÇIKTILARINI HEDEF SÖZLÜKTEKİ KELİMELERLE İLİŞKİLİ SKORLARA DÖNÜŞTÜRÜR.
    decoder_output = decoder_dense(net)

    # DECODER ÇIKTISINI FONKSİYONUN DIŞINA DÖNDÜRÜR.
    return decoder_output


# EĞİTİM İÇİN DECODER'I ENCODER ÇIKTISINI BAŞLANGIÇ DURUMU OLARAK KULLANACAK ŞEKİLDE ÇALIŞTIRIR.
decoder_output = connect_decoder(initial_state=encoder_output)

# ENCODER VE DECODER GİRİŞLERİNİ, DECODER ÇIKTISIYLA BİRLEŞTİREREK EĞİTİM MODELİNİ OLUŞTURUR.
model_train = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

# ENCODER GİRİŞİNİ ALIP ENCODER ÇIKTISINI ÜRETEN AYRI BİR MODEL OLUŞTURUR.
model_encoder = Model(inputs=[encoder_input], outputs=[encoder_output])

# TAHMİN SIRASINDA KULLANILACAK DECODER'I, DECODER GİRİŞİ VE BAŞLANGIÇ DURUMUYLA ÇALIŞTIRIR.
decoder_output = connect_decoder(initial_state=decoder_initial_state)

# DECODER GİRİŞİNİ VE BAŞLANGIÇ GİZLİ DURUMUNU ALIP ÇIKTI ÜRETEN DECODER MODELİNİ OLUŞTURUR.
model_decoder = Model(inputs=[decoder_input, decoder_initial_state], outputs=[decoder_output])


# GERÇEK TOKEN NUMARALARI İLE MODELİN ÜRETTİĞİ SKORLAR ARASINDAKİ KAYBI HESAPLAYAN FONKSİYONU TANIMLAR.
def sparse_cross_entropy(y_true, y_pred):

    # GERÇEK TOKEN NUMARALARINI VE TAHMİN SKORLARINI KULLANARAK HER TOKEN İÇİN KAYIP HESAPLAR.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # HESAPLANAN KAYIP DEĞERLERİNİN ORTALAMASINI ALIR.
    loss_mean = tf.reduce_mean(loss)

    # ORTALAMA KAYIP DEĞERİNİ DÖNDÜRÜR.
    return loss_mean


# MODELİN AĞIRLIKLARINI GÜNCELLEMEK İÇİN RMSPROP OPTİMİZASYON YÖNTEMİNİ BELİRLER.
# lr=1e-3 ÖĞRENME ORANINI 0.001 OLARAK AYARLAR.
optimizer = RMSprop(lr=1e-3)

# DECODER İÇİN GERÇEK HEDEF TOKEN NUMARALARININ VERİLECEĞİ BİR TENSÖR TANIMLAR.
# DTYPE='INT32' DEĞERLERİN TAM SAYI OLDUĞUNU BELİRTİR.
# SHAPE=(NONE, NONE) ÖRNEK SAYISININ VE DİZİ UZUNLUĞUNUN DEĞİŞEBİLECEĞİNİ BELİRTİR.
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

# EĞİTİM MODELİNİ OPTİMİZASYON YÖNTEMİ, KAYIP FONKSİYONU VE HEDEF TENSÖRÜYLE DERLER.
model_train.compile(
    optimizer=optimizer,
    loss=sparse_cross_entropy,
    )

# MODELİN AĞIRLIKLARININ KAYDEDİLECEĞİ DOSYANIN YOLUNU VE ADINI BELİRLER.
path_checkpoint = 'checkpoint.keras'

# HER KAYIT İŞLEMİNDE MODELİN AĞIRLIKLARINI BELİRLENEN DOSYAYA KAYDEDECEK GERİ ÇAĞIRIMI OLUŞTURUR.
# SAVE_WEIGHTS_ONLY=TRUE YALNIZCA MODEL AĞIRLIKLARININ KAYDEDİLMESİNİ SAĞLAR.
checkpoint = ModelCheckpoint(filepath=path_checkpoint, save_weights_only=True)

def sparse_cross_entropy(y_true, y_pred):
    #y_true → Gerçek cevap.
    #y_pred → Modelin tahmini.
    #sparse_cross_entropy → Modelin yaptığı hatayı hesaplayan fonksiyon.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    #TensorFlow, modelin tahminleriyle gerçek cevap arasındaki farkı / hata miktarını hesaplıyor.
    loss_mean = tf.reduce_mean(loss)  # TÜM HATALARIN ORTALAMASINI AL
    return  loss_mean # ORTALAMA HATA DEĞERİNİ GERİ DÖNDÜR

optimizer = RMSprop(learning_rate=1e-3)
# RMSPROP OPTİMİZERINI OLUŞTUR
# ÖĞRENME ORANINI 0.001 OLARAK AYARLA
# RMSprop → Bir optimizasyon algoritması. Modelin hatasını azaltmak için ağırlıkları günceller.
decoder_target = tf.placeholder(dtype='int32', shape=(None,None))
# DECODERIN GERÇEK HEDEF DEĞERLERİNİ TUT
# VERİ TİPİNİ TAM SAYI OLARAK BELİRLE
# BATCH VE ZAMAN BOYUTUNU DEĞİŞKEN BIRAK
#decoder_target, decoder'ın tahminlerini değerlendirmek için kullanılan gerçek cevaplar.
model_train.compile(
    optimizer=optimizer,
    loss=sparse_cross_entropy,
)
# MODELİN EĞİTİM AYARLARINI BELİRLE
# AĞIRLIKLARI GÜNCELLEMEK İÇİN OPTİMİZERİ KULLAN
# HATAYI SPARSE CROSS ENTROPY İLE HESAPLA
# DECODERIN GERÇEK HEDEF VERİSİNİ BELİRLE
path_checkpoint = 'checkpoint.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, save_weights_only=True)
#MODELİN AĞIRLIKLARINI KAYDETMEK İÇİN KULLANILIR
try:
    model_train.load_weights(path_checkpoint) #MODELİN AĞIRLIKLARINI YÜKLEMEK İÇİN KULLANILIR
except Exception as error:
    print('Checkpoint yüklenirken hata oluştu. Eğitime sıfırdan başlanıyor.')
    #MODELİN AĞIRLIKLARI YÜKLENEMEDİĞİ İÇİN EĞİTİME SIFIRDAN BAŞLANIYOR
    print(error)
x_data = {'encoder_input': encoder_input_data, 'decoder_input': decoder_input_data}
#MODELE VERİLEN İNPUTLARIN SÖZLÜK HALİNDE TANIMLANMASI
y_data = {'decoder_output': decoder_output_data}
#MODELDEN BEKLENİLEN OUTPUTLARIN SÖZLÜK HALİNDE TANIMLANMASI
model_train.fit(x=x_data, #MODELE VERİLEN İNPUTLAR
                y=y_data, #MODELDEN BEKLENİLEN OUTPUTLAR
                batch_size=256,#modelin eğitimi sırasında kaç örnek kullanılacağını belirler.
                #Bu örnek sayısı, modelin ağırlıklarını güncellemek için kullanılan veri miktarını ifade eder.
                epochs=1, #modelin eğitimi sırasında tüm eğitim verisinin kaç kez kullanılacağını belirler.
                #Her epoch, modelin tüm eğitim verisi üzerinde bir kez çalıştırılmasını ifade eder.
                callbacks=[checkpoint]) # modelin eğitimi sırasında belirli olaylar gerçekleştiğinde çağrılacak geri arama işlevlerini içeren bir liste.
                #Bu örnekte, modelin ağırlıklarını kaydetmek için kullanılan ModelCheckpoint geri araması kullanılır.
def translate(input_text, true_output_text=None):
    # GİRDİ METNİNİ TOKENLARA DÖNÜŞTÜR VE ÖNCEKİNDEN DOLGU UYGULA
    input_tokens = tokenizer_src.text_to_tokens(
        text=input_text,
        reverse=True,
        padding='pre'
    )

    # ENCODER İLE GİRDİ METNİNİN BAĞLAM BİLGİSİNİ HESAPLA
    initial_state = model_encoder.predict(input_tokens)

    # HEDEF DİLDEKİ MAKSİMUM TOKEN SAYISINI AL
    max_tokens = tokenizer_dest.max_tokens

    # DECODER GİRDİSİ İÇİN SIFIRLARDAN OLUŞAN DİZİ OLUŞTUR
    decoder_input_data = np.zeros((1, max_tokens), dtype='int32')

    # BAŞLANGIÇ TOKENINI AYARLA
    token_int = token_start
    output_text = ''
    count_tokens = 0

    # BİTİŞ TOKENI GELENE VEYA MAKSİMUM UZUNLUĞA ULAŞILANA KADAR DEVAM ET
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int

        # DECODER'A BAŞLANGIÇ DURUMUNU VE TOKEN GİRDİSİNİ VER
        x_data = {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }

        # DECODER'IN TOKEN TAHMİNLERİNİ HESAPLA
        decoder_output = model_decoder.predict(x_data, verbose=0)

        # MEVCUT KONUMDAKİ TOKEN TAHMİNLERİNİ AL
        token_onehot = decoder_output[0, count_tokens, :]

        # EN YÜKSEK SKORLU TOKENIN NUMARASINI BUL
        token_int = np.argmax(token_onehot)

        # TOKEN NUMARASINI KELİMEYE DÖNÜŞTÜR
        sampled_word = tokenizer_dest.token_to_word(token_int)

        # TAHMİN EDİLEN KELİMEYİ ÇEVİRİYE EKLE
        output_text += ' ' + sampled_word
        count_tokens += 1

    # GİRDİ METNİNİ EKRANA YAZDIR
    print('Input text:')
    print(input_text)
    print()

    # ÜRETİLEN ÇEVİRİYİ EKRANA YAZDIR
    print('Translated text:')
    print(output_text.strip())
    print()

    # GERÇEK ÇEVİRİ VERİLMİŞSE ONU DA EKRANA YAZDIR
    if true_output_text is not None:
        print('True output text:')
        print(true_output_text)
        print()
"""1. Girdi metni alınır
input_text, çevirmek istediğin cümledir. Tokenizer bunu modelin anlayabileceği sayılara dönüştürür.
2. Encoder metni işler
model_encoder.predict(input_tokens) girdinin bağlamını temsil eden bilgiyi üretir. Bu bilgi decoder'a aktarılır.
3. Decoder kelimeleri sırayla üretir
while döngüsü her adımda bir sonraki tokenı tahmin eder. Tahmin edilen token bir sonraki adıma girdi olur.
4. Çeviri tamamlanır
Model token_end değerini üretince veya maksimum token sayısına ulaşınca döngü durur ve çeviri yazdırılır."""