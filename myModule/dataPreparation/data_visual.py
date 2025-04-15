import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import myModule.dataPreparation.preprocessing as prepro
import myModule.dataPreparation.splitdata as splitdata
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def report_dataset_final(dataset,kolom_ulasan,kolom_label,file_name):

    st.subheader(f'Tabel dataset ulasan aplikasi {file_name}')

    dataset["at"] = pd.to_datetime(dataset["at"])
    tanggal_terkecil = dataset['at'].min().strftime('%Y-%m-%d %H:%M:%S')
    tanggal_terbesar = dataset['at'].max().strftime('%Y-%m-%d %H:%M:%S')
    
    st.title(f'Dataset ulasan aplikasi {file_name}')
    st.write(f'Dataset ulasan aplikasi {file_name} didapatkan dari scrapping pada google play store dengan jarak data yang diambil pada tanggal {tanggal_terkecil} hingga {tanggal_terbesar} dengan jumlah ulasan sebanyak {len(dataset)}')

    # Hitung jumlah kelas dataset
    st.write("Jumlah kelas sentimen:  ")
    kelas_sentimen = dataset[kolom_label].value_counts()
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
    filtered_df = dataset[dataset[kolom_label].isin(selected_sentiment)]
    st.dataframe(filtered_df)

    # Pilihan time frame
    time_frame = st.radio("Pilih Time Frame", ["Harian", "Bulanan", "Tahunan"])

    # Agregasi berdasarkan pilihan pengguna
    if time_frame == "Harian":
        dataset_grouped = filtered_df.groupby(["at", kolom_label]).size().reset_index(name="jumlah")
    elif time_frame == "Bulanan":
        filtered_df["bulan"] = filtered_df["at"].dt.to_period("M")
        dataset_grouped = filtered_df.groupby(["bulan", kolom_label]).size().reset_index(name="jumlah")
        dataset_grouped["bulan"] = dataset_grouped["bulan"].astype(str)  # Konversi ke string agar terbaca di plot
    else:  # Tahunan
        filtered_df["tahun"] = filtered_df["at"].dt.to_period("Y")
        dataset_grouped = filtered_df.groupby(["tahun", kolom_label]).size().reset_index(name="jumlah")
        dataset_grouped["tahun"] = dataset_grouped["tahun"].astype(str)

    # Warna sesuai dengan sentimen
    sentiment_colors = {"positif": "blue", "netral": "green", "negatif": "red"}

    # Membuat line chart dengan garis putus-putus dan marker
    fig = px.line(
        dataset_grouped,
        x=dataset_grouped.columns[0],  # Bisa 'at', 'bulan', atau 'tahun' tergantung pilihan
        y="jumlah",
        color=kolom_label,
        title=f"Tren Sentimen ({time_frame})",
        color_discrete_map=sentiment_colors,
        markers=True,  # Menampilkan titik data
        line_dash=kolom_label  # Membuat garis putus-putus berdasarkan kategori sentimen
    )

    st.plotly_chart(fig)

    # Bar Chart: Rating Distribution
    fig_bar = px.histogram(filtered_df, x="score", title="Distribusi Rating Aplikasi", nbins=5, color=kolom_label,
                        color_discrete_map={"positif": "blue", "negatif": "red", "netral": "green"},barmode="group")
    st.plotly_chart(fig_bar)

    # Word Cloud for each sentiment
    for sentiment, color in zip(["positif", "negatif", "netral"], ["blue", "red", "green"]):
        sentiment_data = filtered_df[filtered_df[kolom_label] == sentiment]
        if not sentiment_data.empty:
            text = " ".join(sentiment_data["Stopword Removal"].dropna().astype(str))
            if text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                st.subheader(f"Word Cloud - {sentiment.capitalize()}")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
    
