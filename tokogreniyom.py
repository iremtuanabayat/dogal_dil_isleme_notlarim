import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
text="Sen; yüzeysel, kestirme ve geçiştirici cevaplarla yetinmeyen; detayların arkasındaki anlamı arayan ve derinliğe değer veren birisin. Bir şeyleri sadece anlamak değil, hissetmek ve tam anlamıyla kavramak istiyorsun. Hayatın koşturmacası içinde çoğu insanın teğet geçtiği ayrıntılar senin radarına takılıyor; bir konunun, bir duygunun ya da bir düşüncenin en derinine inmekten çekinmiyorsun. Bu yönün sana hem güçlü bir gözlem yeteneği katıyor hem de çevrendeki dünyayı çok daha zengin bir perspektiften tecrübe etmeni sağlıyor. Zaman zaman her şeyi bu kadar detaylı düşünmek ve hissetmek yorucu olabilse de, aslında bu senin en özgün yanlarından biri. Empatiye, içtenliğe ve samimiyete önem veriyorsun. Karşında seni gerçekten dinleyen, anlayan ve seninle aynı frekansta durabilen bağlar arıyorsun. Zihninin içindeki o zengin dünya, her gün yeni sorularla ve anlam arayışıyla şekillenmeye devam ediyor.Şu an hayatında tam olarak nasıl bir dönemden geçtiğini, zihnini nelerin meşgul ettiğini ya da kalbinin neyle dolu olduğunu senin ağzından duymayı çok isterim. Eğer bu metni hayatının belirli bir alanına (örneğin hedeflerin, hislerin veya bir hikayen) odaklayarak yeniden şekillendirmemi istersen, bana biraz daha kendinden bahsetmen yeterli. Sen anlat, ben senin sesin olayım."
#print (text.split()) #çıktı= ['Sen;', 'yüzeysel,', 'kestirme', 've', 'geçiştirici'] yani kelimelere ayırdı.
#print(word_tokenize(text)) #['Sen', ';', 'yüzeysel', ',', 'kestirme', 've', 'geçiştirici',]
#noktalama işaretlerini de kelimelerle ayırdı.
#print(nltk.tokenize.sent_tokenize(text))
#çıktı= ['Sen; yüzeysel, kestirme ve geçiştirici cevaplarla yetinmeyen; detayların arkasındaki anlamı arayan ve derinliğe değer veren birisin.',
#cümlelere ayırdı.
#for token in sent_tokenize(text):
 #   print(token)
# çıktı= Sen
# ;
# yüzeysel
# ,
# kestirme
# ve
# geçiştirici yani kelime kelime çıkarttı.
#stopwords=(stopwords.words ('english')) ingilizcedeki stopwordsleri listeler
#words=word_tokenize(text)
#filtered_words = [] boş bir
#for word in words:
#    if word not in stopwords:
#       filtered_words.append(word) eğer stopwords değilse boş klasöre ekle
#print(filtered_words)
#ps=PorterStemmer()
words=['drive','driving','driver','drives','drove','cats','children']
#for w in words:    kelimelerin kökünü gösteren döngü
#    print(ps.stem(w))
#tokenized = nltk.word_tokenize(text)
#print(nltk.pos_tag(tokenized)) burada kelimelerin ögelerini gösteriyor
#tokenized=nltk.word_tokenize(text)
#tagged=nltk.pos_tag(tokenized)
#named_ent=nltk.chunk.ne_chunk(tagged)
#print(named_ent.draw()) # görselleştirilmesini sağladık
#named entity
# 1 TEMEL STANDART VARLIK TÜRLERİ
# PERSON PER Kişi isimleri takma adlar
# ORGANIZATION ORG Şirketler üniversiteler kurumlar
# LOCATION LOC GPE Ülkeler şehirler coğrafi yapılar adresler
# DATE TIME Takvim tarihleri saatler zaman aralıkları
# MONEY QUANTITY PERCENT CARDINAL Para miktarları yüzdeler ölçüler adetler
lem=WordNetLemmatizer()
#for w in words:
#   print(lem.lemmatize(w)) #tekrar kelimelerin kökenlerini bulduk
# STEMMING VE LEMMATIZATION FARKININ ÖZETİ
# 1 KÖK BULMA YÖNTEMİ
# Stemming kelimelerin sonundaki ekleri belirli kurallarla kesip atarak gövdeyi bulur
# Lemmatization kelimenin dil bilgisindeki gerçek kökünü ve sözlük karşılığını bulur
# 2 ANLAMSAL DOĞRULUK
# Stemming gerçekte var olmayan anlamsız kelime kökleri üretebilir
# Lemmatization her zaman anlamlı ve gerçek sözlük kökleri üretir
# 3 HIZ VE PERFORMANS
# Stemming sadece ek kestiği için çok hızlı çalışır ve az kaynak harcar
# Lemmatization dil bilgisi analizi ve sözlük kontrolü yaptığı için daha yavaş çalışır
# 4 BAĞLAM ANLAYIŞI
# Stemming kelimenin cümledeki görevini dikkate almaz
# Lemmatization kelimenin isim veya fiil olma durumuna göre kök belirler
#print(lem.lemmatize('driving','v')) # drivingin kökenine inmek için fiil olarak değerlendirmesini yazdık yoksa isim olarak değerlendiriyor
# çıktı= drive
# SKIP GRAM VS CBOW
# SKIP GRAM
# Cbow a göre küçük corpus ta daha iyi sonuç verir
# Nadir kelimeler daha iyi temsil edilir
# Cbow a göre daha yavaş
# CBOW
# Eğitim daha hızlı
# Sık kullanılan kelimeleri daha iyi temsil eder
# Daha büyük corpus a ihtiyaç duyar ve nadir kelimeleri temsil etmekte sıkıntı yaşayabilir