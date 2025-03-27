import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import my_module.dataPreparation.preprocessing as prepro

def output_dataset(dataset,kolom_ulasan,kolom_label):
    X_train, X_test, Y_train, Y_test=prepro.data_spilt(kolom_ulasan,kolom_label)
    # Mengambil tanggal paling kecil dan paling besar
    tanggal_terkecil = datetime.strptime(dataset['at'].min(), '%Y-%m-%d %H:%M:%S')
    tanggal_terbesar = datetime.strptime(dataset['at'].max(), '%Y-%m-%d %H:%M:%S')
    name_app = st.session_state['name_app']
    st.title(f'Dataset ulasan aplikasi {name_app}')
    st.write(f'Dataset ulasan aplikasi {name_app}) didapatkan dari scrapping pada google play store dengan jarak data yang diambil pada tangga {tanggal_terkecil.date()} hingga {tanggal_terbesar.date()} dengan jumlah ulasan sebanyak {len(dataset)}')

    st.write(f'berikut merupakan link pada google play store untuk  aplikasi [{name_app})]({st.session_state["url"]}) ')

    st.subheader(f'Tabel dataset ulasan palikasi {name_app}')
    def filter_sentiment(dataset, selected_sentiment):
        return dataset[dataset['sentimen'].isin(selected_sentiment)]

    sentiment_map = {'positif': 'positif', 'negatif': 'negatif', 'netral': 'netral'}
    selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()))
    filtered_data = filter_sentiment(dataset, selected_sentiment)
    st.dataframe(filtered_data)

    # Hitung jumlah kelas dataset
    st.write("Jumlah kelas sentimen:  ")
    kelas_sentimen = dataset['sentimen'].value_counts()
    # st.write(kelas_sentimen)
    datneg,datnet, datpos  = st.columns(3)
    with datpos:
        st.markdown("Positif")
        st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen[0]}</h1>", unsafe_allow_html=True)
    with datnet:
        st.markdown("Netral")
        st.markdown(f"<h1 style='text-align: center; color: red;'>{kelas_sentimen[2]}</h1>", unsafe_allow_html=True)
    with datneg:
        st.markdown("Negatif")
        st.markdown(f"<h1 style='text-align: center; color: aqua;'>{kelas_sentimen[1]}</h1>", unsafe_allow_html=True)
    #membuat diagram
    data = {'sentimen': ['negatif', 'netral', 'positif'],
    'jumlah': [kelas_sentimen[1], kelas_sentimen[2], kelas_sentimen[0]]}
    datasett = pd.DataFrame(data)
    # Membuat diagram pie interaktif
    fig = px.pie(datasett, values='jumlah', names='sentimen', title='Diagram kelas sentimen')
    st.plotly_chart(fig)

def report_dataset_final(dataset):
    name_app = st.session_state['name_app']

    st.subheader(f'Tabel dataset ulasan palikasi {name_app}')
    def filter_sentiment(dataset, selected_sentiment):
        return dataset[dataset['sentimen'].isin(selected_sentiment)]

    sentiment_map = {'positif': 'positif', 'negatif': 'negatif', 'netral': 'netral'}
    selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()))
    filtered_data = filter_sentiment(dataset, selected_sentiment)
    st.dataframe(filtered_data)

    # Hitung jumlah kelas dataset
    st.write("Jumlah kelas sentimen:  ")
    kelas_sentimen = dataset['sentimen'].value_counts()
    # st.write(kelas_sentimen)
    datneg,datnet, datpos  = st.columns(3)
    with datpos:
        st.markdown("Positif")
        st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen[0]}</h1>", unsafe_allow_html=True)
    with datnet:
        st.markdown("Netral")
        st.markdown(f"<h1 style='text-align: center; color: red;'>{kelas_sentimen[2]}</h1>", unsafe_allow_html=True)
    with datneg:
        st.markdown("Negatif")
        st.markdown(f"<h1 style='text-align: center; color: aqua;'>{kelas_sentimen[1]}</h1>", unsafe_allow_html=True)
    #membuat diagram
    data = {'sentimen': ['negatif', 'netral', 'positif'],
    'jumlah': [kelas_sentimen[1], kelas_sentimen[2], kelas_sentimen[0]]}
    datasett = pd.DataFrame(data)
    # Membuat diagram pie interaktif
    fig = px.pie(datasett, values='jumlah', names='sentimen', title='Diagram kelas sentimen')
    st.plotly_chart(fig)
    with st.expander('pembagian dataset') :
        st.write(f"pembagian dataset dilakukan dengan skala 80:20, dimana 80%  menjadi data training sedangkan 20% menjadi data testing dari total dataset yaitu {len(dataset)}")
        st.write(f'Jumlah data training sebanyak {len(X_train)} data ,data training dapat dilihat pada tabel berikut')
        datatrain=pd.concat([X_train, Y_train], axis=1)
        st.dataframe(datatrain)
        st.write(f'Jumlah data testing sebanyak {len(X_test)} data,data testing dapat dilihat pada tabel berikut')
        datatest=pd.concat([X_test, Y_test], axis=1)
        st.dataframe(datatest)