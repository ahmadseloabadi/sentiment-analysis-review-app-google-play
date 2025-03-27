import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from my_module.reusable.downloadButton import download_data
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
    print("vader labeling")

def inset_labeling(dataset):
    print("vader labeling")