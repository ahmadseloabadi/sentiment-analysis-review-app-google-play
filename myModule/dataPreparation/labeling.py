import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from myModule.reusable.downloadButton import download_data
from textblob import TextBlob
import csv

def translate_to_english(dataset):
    translator = GoogleTranslator(source='auto', target='en')
    dataset['Stopword Removal'] = dataset['Stopword Removal'].astype(str)

    dataset['English_Tweet'] = dataset['Stopword Removal'].apply(
        lambda text: translator.translate(text) if text.strip() else ''
    )
    return dataset

def manual_labeling(dataset):
    count=False
    # Load dataset
    data = pd.DataFrame(dataset)
    st.write(data)
    count_edit=st.number_input('masukan banyak data yang akan dilabeli',step=1,value=len(data)-1,min_value=1,max_value=len(data)-1)
    if st.button("menambahkan kolom sentimen"):
        data['sentimen'] = ''
        
    
    with st.form("my_form"):
        st.write('pastikan penyimpan dengan cara mendownload hasil labeling')        
        # Display the selected rows for editing
        edited_df = data.loc[:count_edit].copy()
        st.write('silahkan labeli data :)')
        edited = st.data_editor(edited_df, num_rows="dynamic",column_config={
        "sentimen": st.column_config.SelectboxColumn(
            "sentimen",
            help="kelas sentimen",
            width="medium",
            options=[
                "positif",
                "netral",
                "negatif",
            ],        
        )},)
        # Every form must have a submit button.
        submitted = st.form_submit_button("selesai")    
        if submitted:
            st.write('terima kasih :)')
            download_data(data,"pelabelan manual")

def vader_labeling(dataset):
    nltk.downloader.download('vader_lexicon')
    # Inisialisasi SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Fungsi untuk mendapatkan label berdasarkan nilai sentimen
    def get_sentiment_label(score):
        if score >= 0.05:
            return 'positif'
        elif score <= -0.05:
            return 'negatif'
        else:
            return 'netral'

    # Membaca file CSV dengan kolom 'ulasan'
    data = translate_to_english(dataset)
    st.write("hasil tranlite")
    st.dataframe(data)
    # Membuat kolom baru untuk menyimpan hasil pelabelan
    data['sentimen'] = ""

    # Melakukan pelabelan pada setiap ulasan
    for index, row in data.iterrows():
        ulasan = row['English_Tweet']
        sentiment_score = sia.polarity_scores(ulasan)['compound']
        label = get_sentiment_label(sentiment_score)
        data.at[index, 'sentimen'] = label
    data=data.drop(columns='English_Tweet')
    st.toast("berhasil melakukan pelabelan data", icon='🎉')
    st.subheader('berikut merupakan tampilan dari pelabelan vader')
    st.dataframe(data)
    download_data(data,"pelabelan vader")

def textblob_labeling(dataset):

    # mengambil nilai subjectivity
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    # mengambil nilai polarity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
    # Melakukan Labelling
    def getSentiment(score):
        if score < 0 :
            return 'negatif'
        elif score == 0 :
            return 'netral'
        else :
            return 'positif'
        
    data = translate_to_english(dataset)
    data['English_Tweet'] = data['English_Tweet'].astype(str)

    data['Subjectivity'] = data['English_Tweet'].apply(getSubjectivity)
    data['Polarity'] = data['English_Tweet'].apply(getPolarity)
    data['Sentiment'] = data['Polarity'].apply(getSentiment)

    data=data.drop(columns='English_Tweet')
    st.toast("berhasil melakukan pelabelan data", icon='🎉')
    st.subheader('berikut merupakan tampilan dari pelabelan textblob')
    st.dataframe(data)
    download_data(data,"pelabelan textblob")


def inset_labeling(dataset):
    lexicon_positive = dict()
    with open('data/kamus/positive.tsv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if "word" not in row and "weight" not in row:
                lexicon_positive[row[0]] = int(row[1])

    lexicon_negative = dict()
    with open('data/kamus/negative.tsv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if "word" not in row and "weight" not in row:
                lexicon_negative[row[0]] = int(row[1])

    def inset_lexicon(text):
        word_scores = {}
        score= 0
        for word in text:
            if word in lexicon_positive:
                score += lexicon_positive[word]
                word_scores[word] = lexicon_positive[word]
            if word in lexicon_negative:
                score += lexicon_negative[word]
                word_scores[word] =lexicon_negative[word]
            if (word in lexicon_negative) and (word in lexicon_positive):
                word_scores[word] = lexicon_negative[word] + lexicon_positive[word]
        polarity=''
        if (score > 0):
            polarity = 'positif'
        elif(score<0):
            polarity = 'negatif'
        else:
            polarity = 'netral'
        return word_scores, score, polarity
    
    results = dataset.apply(inset_lexicon)
    results = list(zip(*results))
    # Menambahkan kolom baru ke DataFrame
    dataset['score_term'] = results[0]
    dataset['polarity_score'] = results[1]
    dataset['sentiment'] = results[2]
    dataset=dataset[['content','score_term','polarity_score','sentiment']]
    st.toast("berhasil melakukan pelabelan data", icon='🎉')
    st.subheader('berikut merupakan tampilan dari pelabelan inset lexicon')
    st.dataframe(dataset)
    download_data(dataset,"pelabelan inset lexicon")