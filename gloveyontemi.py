from gensim.models import KeyedVectors
# normalde glove dosyaları keyedvectors için dönüştürülürmüş ancak güncelemeyle artık dönüştürülmüyor
# GloVe dosyasını dönüştürmeden doğrudan yükleme
model = KeyedVectors.load_word2vec_format(
    "glove.6B.100d.txt",
    binary=False,
    no_header=True  # GloVe dosyalarında ilk satırda başlık olmadığını belirtir
)

print("GloVe dosyası dönüştürülmeden doğrudan yüklendi!")