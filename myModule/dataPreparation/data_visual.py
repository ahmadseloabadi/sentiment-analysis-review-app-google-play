import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import myModule.dataPreparation.preprocessing as prepro
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def output_dataset(dataset,file_name,kolom_ulasan):
    # Mengambil tanggal paling kecil dan paling besar
    tanggal_terkecil = datetime.strptime(dataset['at'].min(), '%Y-%m-%d %H:%M:%S')
    tanggal_terbesar = datetime.strptime(dataset['at'].max(), '%Y-%m-%d %H:%M:%S')
    
    st.title(f'Dataset ulasan aplikasi {file_name}')
    st.write(f'Dataset ulasan aplikasi {file_name} didapatkan dari scrapping pada google play store dengan jarak data yang diambil pada tanggal {tanggal_terkecil.date()} hingga {tanggal_terbesar.date()} dengan jumlah ulasan sebanyak {len(dataset)}')

    st.subheader(f'Tabel dataset ulasan palikasi {file_name}')
    def filter_sentiment(dataset, selected_sentiment):
        return dataset[dataset[kolom_ulasan].isin(selected_sentiment)]

    sentiment_map = {'positif': 'positif', 'negatif': 'negatif', 'netral': 'netral'}
    selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()))
    filtered_data = filter_sentiment(dataset, selected_sentiment)
    st.dataframe(filtered_data)

    # Hitung jumlah kelas dataset
    st.write("Jumlah kelas sentimen:  ")
    kelas_sentimen = dataset[kolom_ulasan].value_counts()
    # st.write(kelas_sentimen)
    datneg,datnet, datpos  = st.columns(3)
    with datpos:
        st.markdown("Positif")
        st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen['positif']}</h1>", unsafe_allow_html=True)
    with datnet:
        st.markdown("Netral")
        st.markdown(f"<h1 style='text-align: center; color: green;'>{kelas_sentimen['netral']}</h1>", unsafe_allow_html=True)
    with datneg:
        st.markdown("Negatif")
        st.markdown(f"<h1 style='text-align: center; color: red;'>{kelas_sentimen['negatif']}</h1>", unsafe_allow_html=True)
    #membuat diagram
    data = {kolom_ulasan: ['negatif', 'netral', 'positif'],
    'jumlah': [kelas_sentimen[1], kelas_sentimen[2], kelas_sentimen[0]]}
    datasett = pd.DataFrame(data)
    # Membuat diagram pie interaktif
    fig = px.pie(datasett, values='jumlah', names=kolom_ulasan, title='Diagram kelas sentimen')
    st.plotly_chart(fig)

def report_dataset_final(dataset,kolom_ulasan,kolom_label,file_name):
    X_train, X_test, Y_train, Y_test=prepro.data_spilt(kolom_ulasan,kolom_label)


    st.subheader(f'Tabel dataset ulasan palikasi {file_name}')

    tanggal_terkecil = datetime.strptime(dataset['at'].min(), '%Y-%m-%d %H:%M:%S')
    tanggal_terbesar = datetime.strptime(dataset['at'].max(), '%Y-%m-%d %H:%M:%S')
    dataset["at"] = pd.to_datetime(dataset["at"])
    
    st.title(f'Dataset ulasan aplikasi {file_name}')
    st.write(f'Dataset ulasan aplikasi {file_name} didapatkan dari scrapping pada google play store dengan jarak data yang diambil pada tanggal {tanggal_terkecil.date()} hingga {tanggal_terbesar.date()} dengan jumlah ulasan sebanyak {len(dataset)}')

    # Hitung jumlah kelas dataset
    st.write("Jumlah kelas sentimen:  ")
    kelas_sentimen = dataset['sentimen'].value_counts()
    datneg,datnet, datpos  = st.columns(3)
    with datpos:
        st.markdown("Positif")
        st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen['positif']}</h1>", unsafe_allow_html=True)
    with datnet:
        st.markdown("Netral")
        st.markdown(f"<h1 style='text-align: center; color: green;'>{kelas_sentimen['netral']}</h1>", unsafe_allow_html=True)
    with datneg:
        st.markdown("Negatif")
        st.markdown(f"<h1 style='text-align: center; color: red;'>{kelas_sentimen['negatif']}</h1>", unsafe_allow_html=True)
    #membuat diagram
    # Sentiment filter
    sentiment_map = {'positif': 'positif', 'negatif': 'negatif', 'netral': 'netral'}
    selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()))
    filtered_df = dataset[dataset['sentimen'].isin(selected_sentiment)]
    st.dataframe(filtered_df)

    # Pilihan time frame
    time_frame = st.radio("Pilih Time Frame", ["Harian", "Bulanan", "Tahunan"])

    # Agregasi berdasarkan pilihan pengguna
    if time_frame == "Harian":
        
        dataset_grouped = filtered_df.groupby(["at", "sentimen"]).size().reset_index(name="jumlah")
    elif time_frame == "Bulanan":
        filtered_df["bulan"] = filtered_df["at"].dt.to_period("M")
        dataset_grouped = filtered_df.groupby(["bulan", "sentimen"]).size().reset_index(name="jumlah")
        dataset_grouped["bulan"] = dataset_grouped["bulan"].astype(str)  # Konversi ke string agar terbaca di plot
    else:  # Tahunan
        filtered_df["tahun"] = filtered_df["at"].dt.to_period("Y")
        dataset_grouped = filtered_df.groupby(["tahun", "sentimen"]).size().reset_index(name="jumlah")
        dataset_grouped["tahun"] = dataset_grouped["tahun"].astype(str)

    # Warna sesuai dengan sentimen
    sentiment_colors = {"positif": "blue", "netral": "green", "negatif": "red"}

    # Membuat line chart dengan garis putus-putus dan marker
    fig = px.line(
        dataset_grouped,
        x=dataset_grouped.columns[0],  # Bisa 'at', 'bulan', atau 'tahun' tergantung pilihan
        y="jumlah",
        color="sentimen",
        title=f"Tren Sentimen ({time_frame})",
        color_discrete_map=sentiment_colors,
        markers=True,  # Menampilkan titik data
        line_dash="sentimen"  # Membuat garis putus-putus berdasarkan kategori sentimen
    )

    st.plotly_chart(fig)

    # Bar Chart: Rating Distribution
    fig_bar = px.histogram(filtered_df, x="score", title="Distribusi Rating Aplikasi", nbins=5, color="sentimen",
                        color_discrete_map={"positif": "blue", "negatif": "red", "netral": "green"},barmode="group")
    st.plotly_chart(fig_bar)

    # Word Cloud for each sentiment
    for sentiment, color in zip(["positif", "negatif", "netral"], ["blue", "red", "green"]):
        sentiment_data = filtered_df[filtered_df["sentimen"] == sentiment]
        if not sentiment_data.empty:
            text = " ".join(sentiment_data["Stopword Removal"].dropna().astype(str))
            if text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                st.subheader(f"Word Cloud - {sentiment.capitalize()}")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
    with st.expander('pembagian dataset') :
        st.write(f"pembagian dataset dilakukan dengan skala 80:20, dimana 80%  menjadi data training sedangkan 20% menjadi data testing dari total dataset yaitu {len(dataset)}")
        st.write(f'Jumlah data training sebanyak {len(X_train)} data ,data training dapat dilihat pada tabel berikut')
        datatrain=pd.concat([X_train, Y_train], axis=1)
        st.dataframe(datatrain)
        st.write(f'Jumlah data testing sebanyak {len(X_test)} data,data testing dapat dilihat pada tabel berikut')
        datatest=pd.concat([X_test, Y_test], axis=1)
        st.dataframe(datatest)