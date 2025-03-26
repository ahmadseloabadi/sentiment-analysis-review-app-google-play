import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.preprocessing import LabelEncoder

vectorizer = TfidfVectorizer()
Encoder = LabelEncoder()

# text preprosessing
def cleansing(kalimat_baru): 
    kalimat_baru = re.sub(r'@[A-Za-a0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r'#[A-Za-z0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"http\S+",' ',kalimat_baru)
    kalimat_baru = re.sub(r'[0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", kalimat_baru)
    kalimat_baru = re.sub(r"\b[a-zA-Z]\b", " ", kalimat_baru)
    kalimat_baru = kalimat_baru.strip(' ')
    # menghilangkan emoji
    def clearEmoji(ulasan):
        return ulasan.encode('ascii', 'ignore').decode('ascii')
    kalimat_baru =clearEmoji(kalimat_baru)
    def replaceTOM(ulasan):
        pola = re.compile(r'(.)\1{2,}', re.DOTALL)
        return pola.sub(r'\1', ulasan)
    kalimat_baru=replaceTOM(kalimat_baru)
    return kalimat_baru
def casefolding(kalimat_baru):
    kalimat_baru = kalimat_baru.lower()
    return kalimat_baru
def tokenizing(kalimat_baru):
    kalimat_baru = word_tokenize(kalimat_baru)
    return kalimat_baru
def slangword (kalimat_baru):
    kamusSlang = eval(open("data/dictionary/slangwords.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
    content = []
    for kata in kalimat_baru:
        filter_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata.lower())
        if filter_slang.startswith('tidak_'):
          kata_depan = 'tidak_'
          kata_belakang = kata[6:]
          kata_belakang_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata_belakang.lower())
          kata_hasil = kata_depan + kata_belakang_slang
          content.append(kata_hasil)
        else:
          content.append(filter_slang)
    kalimat_baru = content
    return kalimat_baru
def handle_negation(kalimat_baru):
    negation_words = ["tidak", "bukan", "tak", "tiada", "jangan", "gak",'ga']
    new_words = []
    prev_word_is_negation = False
    for word in kalimat_baru:
        if word in negation_words:
            new_words.append("tidak_")
            prev_word_is_negation = True
        elif prev_word_is_negation:
            new_words[-1] += word
            prev_word_is_negation = False
        else:
            new_words.append(word)
    return new_words
def stopword (kalimat_baru):
    daftar_stopword = stopwords.words('indonesian')
    daftar_stopword.extend(["yg", "dg", "rt", "dgn", "ny", "d",'gb','ahk','g','anjing','ga','gua','nder']) 
    # Membaca file teks stopword menggunakan pandas
    txt_stopword = pd.read_csv("data/dictionary/stopwords.txt", names=["stopwords"], header=None)

    # Menggabungkan daftar stopword dari NLTK dengan daftar stopword dari file teks
    daftar_stopword.extend(txt_stopword['stopwords'].tolist())

    # Mengubah daftar stopword menjadi set untuk pencarian yang lebih efisien
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        cleaned_words = []
        for word in words:
            # Memisahkan kata dengan tambahan "tidak_"
            if word.startswith("tidak_"):
                cleaned_words.append(word[:5])
                cleaned_words.append(word[6:])
            elif word not in daftar_stopword:
                cleaned_words.append(word)
        return cleaned_words
    kalimat_baru = stopwordText(kalimat_baru)
    return kalimat_baru 
def stemming(kalimat_baru):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # Lakukan stemming pada setiap kata
    stemmed_words = [stemmer.stem(word) for word in kalimat_baru]
    return stemmed_words

def output_tfidf(dataset):
    # Create CountVectorizer instance
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(dataset)

    # Create TfidfTransformer instance
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_count)

    # Create TfidfVectorizer instance
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf_vectorized = tfidf_vectorizer.fit_transform(dataset)

    # Get the feature names from CountVectorizer or TfidfVectorizer
    feature_names = count_vectorizer.get_feature_names_out()  # or tfidf_vectorizer.get_feature_names()

    # Create a dictionary to store the results
    results = {"Ulasan": [], "Term": [], "TF": [], "IDF": [], "TF-IDF": []}

    # Loop over the documents
    for i in range(len(dataset)):
        # Add the document to the results dictionary
        results["Ulasan"].extend([f" ulasan{i+1}"] * len(feature_names))
        # Add the feature names to the results dictionary
        results["Term"].extend(feature_names)
        # Calculate the TF, IDF, and TF-IDF for each feature in the document
        for j, feature in enumerate(feature_names):
            tf = X_count[i, j]
            idf = tfidf_transformer.idf_[j]  # or X_tfidf_vectorized.idf_[j]
            tf_idf_score = X_tfidf[i, j]  # or X_tfidf_vectorized[i, j]
            # Add the results to the dictionary
            results["TF"].append(tf)
            results["IDF"].append(idf)
            results["TF-IDF"].append(tf_idf_score)
    # Convert the results dictionary to a Pandas dataframe
    df = pd.DataFrame(results)

    #filter nilai term
    newdf = df[(df.TF != 0 )]
    # Save the results to a CSV file
    return newdf

def data_spilt(kolom_ulasan,kolom_label):
    x=kolom_ulasan
    y=kolom_label
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20)
    return X_train, X_test, Y_train, Y_test

def data_tfidf(X_train, X_test, Y_train, Y_test):
    
    x_train = vectorizer.fit_transform(X_train)
    x_test = vectorizer.transform(X_test)
    
    y_train = Encoder.fit_transform(Y_train)
    y_test = Encoder.fit_transform(Y_test)
    return y_train, y_test,x_train, x_test