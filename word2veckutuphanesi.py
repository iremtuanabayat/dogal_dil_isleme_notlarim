import numpy as np
import string
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. DOSYA OKUMA VE TEMİZLEME
with open("hurriyet.txt", "r", encoding="utf-8") as f:
    text = f.read()

t_list = text.split('\n')

corpus = []
for cumle in t_list:
    # Küçük harfe çevirip noktalama işaretlerini temizleme
    cumle_temiz = cumle.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = cumle_temiz.split()
    if tokens:
        corpus.append(tokens)

# 2. MODEL EĞİTİMİ
model = word2vec.Word2Vec(corpus, vector_size=100, window=5, min_count=5, sg=1)


# 3. TSNE VE GÖRSELLEŞTİRME FONKSİYONU
def closestwords_tsneplot(model, word):
    word = word.lower()  # Aramayı küçük harfle eşleme

    if word not in model.wv:
        print(f"HATA: '{word}' kelimesi modelin kelime hazinesinde (vocab) yok!")
        return

    word_labels = [word]
    word_vectors = [model.wv[word]]

    close_words = model.wv.most_similar(word)

    for w, _ in close_words:
        word_labels.append(w)
        word_vectors.append(model.wv[w])

    word_vectors = np.array(word_vectors)

    # perplexity < n_samples (11 eleman olduğu için perplexity=3 verdik)
    tsne = TSNE(n_components=2, perplexity=3, random_state=0)
    Y = tsne.fit_transform(word_vectors)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(x_coords, y_coords, color='blue', edgecolors='k')

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(5, -2), textcoords='offset points', fontsize=11)

    plt.title(f"'{word}' Kelimesi için t-SNE Görselleştirmesi")
    plt.grid(True)
    plt.show()


# 4. FONKSİYONU ÇAĞIRMA (Lütfen metinde geçtiğinden emin olduğunuz bir kelime girin)
closestwords_tsneplot(model, "berlin")