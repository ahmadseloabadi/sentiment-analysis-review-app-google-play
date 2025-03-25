#import library
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import re
import numpy as np
import random as rd
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

import Genetic_Algorithm as svm_hp_opt


from google_play_scraper import Sort, reviews_all, app
from google_play_scraper.exceptions import NotFoundError



# Set page layout and title
st.set_page_config(page_title="review google play app")

vectorizer = TfidfVectorizer()
Encoder = LabelEncoder()


def extract_app_details(url):
    """Ekstrak app ID, bahasa, dan negara dari URL Play Store."""
    app_id_match = re.search(r'id=([^&]+)', url)
    lang_match = re.search(r'hl=([^&]+)', url)
    country_match = re.search(r'gl=([^&]+)', url)
    
    app_id = app_id_match.group(1) if app_id_match else None
    lang = lang_match.group(1) if lang_match else 'id'  # Default ke Inggris jika tidak ada
    country = country_match.group(1) if country_match else 'ID'  # Default ke US jika tidak ada
    
    return app_id, lang, country
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

if 'is_scrap' not in st.session_state and st.session_state['is_scraping'] is not True:
    st.toast("silahkan lakukan scraping data terlebih dahulu")
#scrapping google play store
def scrapping_play_store(url_app):
    try :
        app_id, lang, country=extract_app_details(url_app)
        with st.spinner("Sedang mengambil data..."):
            infoapp = app(
            app_id,
            lang=lang, # defaults to 'en'
            country=country # defaults to 'us'
            )
            name_app=infoapp.get('title')
            st.session_state['name_app']=name_app
            st.toast(f"sedang melakukan scraping data aplikasi {st.session_state['name_app']}")
            
            scrapping = reviews_all(
            app_id,
            sleep_milliseconds=0, # defaults to 0
            lang=lang, # defaults to 'en'
            country=country, # defaults to 'us'
            sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT , you can use Sort.NEWEST to get newst reviews
            )
            
        df_scrapping = pd.DataFrame(np.array(scrapping),columns=['review'])
        df_scrapping = df_scrapping.join(pd.DataFrame(df_scrapping.pop('review').tolist()))
        new_df = df_scrapping[['userName', 'score','at', 'content']]
        sorted_df = new_df.sort_values(by='at', ascending=False)
        ulasan = sorted_df
        ulasan.dropna()
        st.session_state['url']=url_app
        st.session_state['is_scraping'] = True
    except NotFoundError:
        st.error("⚠️ Aplikasi tidak ditemukan. Pastikan URL yang dimasukkan benar!")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        
    else:
        st.toast(f"data aplikasi {name_app} berhasil diambil")
        # Mengambil tanggal paling kecil dan paling besar
        tanggal_terkecil = sorted_df['at'].min().strftime('%Y-%m-%d %H:%M:%S')
        tanggal_terbesar = sorted_df['at'].max().strftime('%Y-%m-%d %H:%M:%S')
        
        st.write(f"hasil scraping ulasan aplikasi {name_app} pada google play store dengan jarak data yang diambil pada tangga {tanggal_terkecil} hingga {tanggal_terbesar} dengan jumlah ulasan sebanyak {len(sorted_df)}")
        sorted_df = sorted_df.sort_index()
        st.dataframe(sorted_df)
        
        st.write("silahkan download data untuk proses selanjutnya :)")
        csv = convert_for_download(sorted_df)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ulasan aplikasi {name_app}.csv",
            mime="text/csv",
            icon=":material/download:",
        )

def create_sentiment_column(dataset_path):
    count=False
    # Load dataset
    data = pd.read_csv(dataset_path)
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

def model_svm(C,gamma,x_train,y_train):
    if C is not None:
        modelsvm = svm.SVC(C=C,gamma=gamma)
    else:
        modelsvm = svm.SVC()
    modelsvm.fit(x_train, y_train)
    return modelsvm
def output_dataset(dataset,kolom_ulasan,kolom_label):
    X_train, X_test, Y_train, Y_test=data_spilt(kolom_ulasan,kolom_label)
    # Mengambil tanggal paling kecil dan paling besar
    tanggal_terkecil = datetime.strptime(dataset['at'].min(), '%Y-%m-%d %H:%M:%S')
    tanggal_terbesar = datetime.strptime(dataset['at'].max(), '%Y-%m-%d %H:%M:%S')
    name_app = st.session_state['name_app']
    st.title(f'Dataset ulasan aplikasi {name_app}')
    st.write(f'Dataset ulasan aplikasi {name_app}) didapatkan dari scrapping pada google play store dengan jarak data yang diambil pada tangga {tanggal_terkecil.date()} hingga {tanggal_terbesar.date()} dengan jumlah ulasan sebanyak {len(dataset)}')

    st.write('berikut merupakan link pada google play store untuk  aplikasi [{name_app})](https://play.google.com/store/apps/details?id=com.finaccel.android&hl=id&showAllReviews=true) ')

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


kfold=5
def data_weight_split(dataset):
    kolom_ulasan =dataset['Stopword Removal']
    kolom_label =dataset['sentimen']
    X_train, X_test, Y_train, Y_test=data_spilt(kolom_ulasan,kolom_label)
    y_train, y_test,x_train, x_test=data_tfidf(X_train, X_test, Y_train, Y_test)
    return y_train, y_test,x_train, x_test

def class_report(model,x_test,y_test):
    # Using the model to predict the labels of the test data
    y_pred = model.predict(x_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write('plot confusion matrix')
    # Membuat plot matriks kebingungan
    f, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot(f)
#side bar
with st.sidebar :
    selected = option_menu('sentimen analisis',['Home','Text preprocessing','Algoritma Genetika','Testing','Report'])

if(selected == 'Home') :
    tab1,tab2=st.tabs(['Main','Scrapping'])
    with tab1 :
        st.title('OPTIMASI PARAMETER METODE SUPPORT VECTOR MACHINE DENGAN ALGORITMA GENETIKA PADA ULASAN APLIKASI PADA GOOGLE PLAY STORE ')
        st.subheader('support vector machine')
        st.write('Support Vector Machine (SVM) merupakan algoritma terbaik diantara beberapa algoritma lainnya seperti Naïve Bayes, Decision Trees, dan Random Forest karena mampu mengkomputasi data dengan dimensi tinggi sehingga tingkat akurasi yang dihasilkan lebih baik (Andrean, 2024). Metode SVM juga memiliki kemampuan untuk mengatasi overfitting dan tidak membutuhkan data yang terlalu besar dengan hasil akurasi tinggi serta tingkat kesalahan yang relatif kecil (Harafani & Maulana, 2019) Selain banyaknya kelebihan yang dimiliki, metode SVM masih memiliki kelemahan yaitu sulitnya menentukan parameter yang optimal (Istiadi & Rahman, 2020). Jika menggunakan parameter default hasil akurasi dan klasifikasi SVM tidak akan semaksimal apabila menggunakan pemilihan parameter yang tepat.')
        st.image('img/svm.png')
        st.subheader('Algoritma Genetika')
        st.write('Algoritma Genetika adalah metode heuristik yang dikembangkan berdasarkan prinsip-prinsip genetika dan proses seleksi alam teori evolusi Darwin. Algoritma Genetika memiliki metode yang dapat menangani masalah optimasi nonlinier yang berdimensi tinggi dan sangat berguna untuk diimplementasikan pada saat range terbaik dari parameter SVM sama sekali tidak diketahui. Algoritma Genetika dapat menghasilkan parameter SVM yang optimal secara bersamaan. Tujuan utama algoritma ini yaitu mendapatkan populasi baru yang lebih baik dibandingkan populasi sebelumnya (Kusumaningrum, 2017).')
        st.image('img/GA.jpg')
    with tab2:
        st.write(f'scrapping data ulasan aplikasi pada google play store')
        url_app=st.text_input("masukan link aplikasi google playstore :")
        if st.button('start scrapping') :
            scrapping_result=scrapping_play_store(url_app)
            
            
            
elif(selected == 'Text preprocessing') :
    tab1,tab2,tab3=st.tabs(['Labeling','Dataset','Text preprosesing'])

            
    with tab1 :
        file_labeling = st.file_uploader("masukan data yang akan dilabeli", key="labeling", type='csv')
        if file_labeling is not None:
            create_sentiment_column(file_labeling)
    
    with tab2 :        
        file_labeling = st.file_uploader("masukan data yg sudah dilabeli", key="datasettt", type='csv')
        if file_labeling is not None:
            file = pd.read_csv(file_labeling)
            st.dataframe(file)
            ulasan = st.text_input('masukan nama kolom ulasan data',value="content")
            label = st.text_input('masukan nama kolom labeling data',value="sentimen")
            if st.button("tampilkan dataset"):
                kolom_ulasan =file[ulasan]
                kolom_label =file[label]
                output_dataset(file,kolom_ulasan,kolom_label)
    with tab3 :

        uploaded_file = st.file_uploader("masukan data yang akan dilakukan preprocessing", key="preprocewssing", type='csv')

        if uploaded_file is not None:

            dataset = pd.read_csv(uploaded_file)
            st.write('proses untuk menampilkan hasil text preprocessing dan tf-idf mungkin membutuhkan waktu yang cukup lama')
            kolom = st.text_input('masukan nama kolom ulasan/review pada data yang di input',value="content")
            st.write('tampilan dataset yang di input')
            st.write(dataset)

            if st.button('start text preprocessing') :
                dataset['Cleansing']= dataset[kolom].apply(cleansing)
                st.write('tampilan hasil cleansing')
                cleansing = dataset[[kolom,'Cleansing']]
                st.dataframe(cleansing)

                dataset['CaseFolding']= dataset['Cleansing'].apply(casefolding)
                st.write('tampilan hasil casefolding')
                casefolding= dataset[['Cleansing','CaseFolding']]
                st.dataframe(casefolding)
                
                dataset['Tokenizing']= dataset['CaseFolding'].apply(tokenizing)
                st.write('tampilan hasil tokenizing')
                tokenizing= dataset[['CaseFolding','Tokenizing']]
                st.dataframe(tokenizing)
                
                dataset['stemming']= dataset['Tokenizing'].apply(stemming)
                st.write('tampilan hasil stemming')
                stemming= dataset[['Tokenizing','stemming']]
                st.dataframe(stemming)

                dataset['negasi']= dataset['stemming'].apply(handle_negation)
                st.write('tampilan hasil negasi')
                negasi= dataset[['stemming','negasi']]
                st.dataframe(negasi)

                dataset['wordnormalization']= dataset['negasi'].apply(slangword)
                st.write('tampilan hasil word normalization')
                wordnormalization= dataset[['negasi','wordnormalization']]
                st.dataframe(wordnormalization)

                dataset['stopword']= dataset['wordnormalization'].apply(stopword)
                st.write('tampilan hasil stopword')
                stopword= dataset[['wordnormalization','stopword']]
                #merubah list ke str 
                dataset['Stopword Removal'] = dataset['stopword'].apply(' '.join)
                st.dataframe(stopword)
                dataset.to_csv('hasil_preprocessing.csv',index=False)
                prepro=True
                
            if st.button('start pembobotan TF-IDF') :
                dataset=pd.read_csv('hasil_preprocessing.csv')
                datasett=dataset['Stopword Removal'] 
                start_tfidf=output_tfidf(datasett)
                st.write('tampilan hasil pembobotan TF-IDF')
                st.dataframe(start_tfidf,use_container_width=True)
                # Save the results to a CSV file
                    
elif(selected == 'Modeling') :
    tab1,tab2=st.tabs(['Algoritma Genetika','Testing'])
    with tab1:
        st.header("Algoritma Genetika")
        probcros = st.number_input('probabilitas crossover',format="%0.1f",value=0.6,step=0.1)
        probmutasi = st.number_input('probabilitas mutasi',format="%0.1f",value=0.2,step=0.1)
        populasi = st.number_input('banyak populasi',step=1,value=6)
        generasi = st.number_input('banyak generasi',step=1,value=3)

        if populasi % 2 != 0:  # Cek apakah x tidak habis dibagi 2
            populasi += 1 

        if st.button('algoritma genetika') :
            with st.expander('data generasi') :
                x=x_train
                y=y_train
                prob_crsvr = probcros 
                prob_mutation = probmutasi 
                population = populasi
                generations = generasi 

                kfold = 5

                x_y_string = np.array([0,1,0,0,0,1,0,0,1,0,0,1,
                                    0,1,1,1,0,0,1,0,1,1,1,0]) 


                pool_of_solutions = np.empty((0,len(x_y_string)))

                best_of_a_generation = np.empty((0,len(x_y_string)+1))

                for i in range(population):
                    rd.shuffle(x_y_string)
                    pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))

                gen = 1 
                gene=[]
                c_values = []
                gamma_values = []
                fitness_values = []
                cb_values = []
                gammab_values = []
                fitnessb_values = []
                for i in range(generations): 
                    
                    new_population = np.empty((0,len(x_y_string)))
                    
                    new_population_with_fitness_val = np.empty((0,len(x_y_string)+1))
                    
                    sorted_best = np.empty((0,len(x_y_string)+1))
                    
                    st.write()
                    st.write()
                    st.write("Generasi ke -", gen) 
                    
                    family = 1 
                    
                    for j in range(int(population/2)): 
                        
                        st.write()
                        st.write("populasi ke -", family) 
                        
                        parent_1 = svm_hp_opt.find_parents_ts(pool_of_solutions,x=x,y=y)[0]
                        parent_2 = svm_hp_opt.find_parents_ts(pool_of_solutions,x=x,y=y)[1]
                        
                        child_1 = svm_hp_opt.crossover(parent_1,parent_2,prob_crsvr=prob_crsvr)[0]
                        child_2 = svm_hp_opt.crossover(parent_1,parent_2,prob_crsvr=prob_crsvr)[1]
                        
                        mutated_child_1 = svm_hp_opt.mutation(child_1,child_2, prob_mutation=prob_mutation)[0]
                        mutated_child_2 = svm_hp_opt.mutation(child_1,child_2,prob_mutation=prob_mutation)[1]
                        
                        fitness_val_mutated_child_1 = svm_hp_opt.fitness(x=x,y=y,chromosome=mutated_child_1,kfold=kfold)[2]
                        fitness_val_mutated_child_2 = svm_hp_opt.fitness(x=x,y=y,chromosome=mutated_child_2,kfold=kfold)[2]
                        
                        

                        mutant_1_with_fitness_val = np.hstack((fitness_val_mutated_child_1,mutated_child_1)) 
                        
                        mutant_2_with_fitness_val = np.hstack((fitness_val_mutated_child_2,mutated_child_2)) 
                        
                        
                        new_population = np.vstack((new_population,
                                                    mutated_child_1,
                                                    mutated_child_2))
                        
                        
                        new_population_with_fitness_val = np.vstack((new_population_with_fitness_val,
                                                                mutant_1_with_fitness_val,
                                                                mutant_2_with_fitness_val))
                        
                        st.write(f"Parent 1:",str(parent_1))
                        st.write(f"Parent 2:",str(parent_2))
                        st.write(f"Child 1:",str(child_1))
                        st.write(f"Child 2:",str(child_2))
                        st.write(f"Mutated Child 1:",str(mutated_child_1))
                        st.write(f"Mutated Child 2:",str(mutated_child_2))
                        st.write(f"nilai fitness 1:",fitness_val_mutated_child_1)
                        st.write(f"nilai fitness 2:",fitness_val_mutated_child_2)

                        family = family+1
                    pool_of_solutions = new_population
                    
                    sorted_best = np.array(sorted(new_population_with_fitness_val,
                                                            key=lambda x:x[0]))
                    
                    best_of_a_generation = np.vstack((best_of_a_generation,
                                                    sorted_best[0]))
                    
                    
                    sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                                        key=lambda x:x[0]))

                    gen = gen+1 
                    

                    best_string_convergence = sorted_best[0]
                    best_string_bestvalue = sorted_best_of_a_generation[0]
                    final_solution_convergence = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_convergence[0:],kfold=kfold)
                    final_solution_best = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_bestvalue[0:],kfold=kfold)
                    
                    gene.append(gen-1)

                    c_values.append(final_solution_convergence[0])
                    gamma_values.append(final_solution_convergence[1])
                    fitness_values.append(final_solution_convergence[2])

                    cb_values.append(final_solution_best[0])
                    gammab_values.append(final_solution_best[1])
                    fitnessb_values.append(final_solution_best[2])
                    # create a dictionary to store the data
                    results = {'generasi':gene,
                            'C_con': c_values,
                            'Gamma_con': gamma_values,
                            'Fitness_con': fitness_values,
                            'C_best': cb_values,
                            'Gamma_best': gammab_values,
                            'Fitness_best': fitnessb_values}
                sorted_last_population = np.array(sorted(new_population_with_fitness_val,key=lambda x:x[0]))

                sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,key=lambda x:x[0]))

                sorted_last_population[:,0] = (sorted_last_population[:,0]) # get accuracy instead of error
                sorted_best_of_a_generation[:,0] = (sorted_best_of_a_generation[:,0])
                best_string_convergence = sorted_last_population[0]

                best_string_overall = sorted_best_of_a_generation[0]

                # to decode the x and y chromosomes to their real values
                final_solution_convergence = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_convergence[1:],
                                                                        kfold=kfold)

                final_solution_overall = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_overall[1:],
                                                                    kfold=kfold)

            hasil_pengujian=pd.DataFrame(results)
            st.write(hasil_pengujian)
            gabest1 = hasil_pengujian['C_best'].iloc[-1]
            gabest2 = hasil_pengujian['Gamma_best'].iloc[-1]
            gabest3 = hasil_pengujian['Fitness_best'].iloc[-1]
            st.session_state["bestCbaru"] = gabest1
            st.session_state["bestGammabaru"] = gabest2

            best_convergen = hasil_pengujian.loc[(hasil_pengujian['C_con'] == gabest1) & (hasil_pengujian['Gamma_con'] == gabest2) & (hasil_pengujian['Fitness_con'] == gabest3)]
            akurasi=1-best_convergen['Fitness_con'].values[0]
            st.write(f"Kesimpulan dari tabel di atas yaitu : Algoritma Genetika terbaik Pada Generasi { best_convergen['generasi'].values[0]} denngan nilai C :{round(best_convergen['C_con'].values[0],4)} ,nilai Gamma {round(best_convergen['Gamma_con'].values[0],4)},dan nilai Fitness {best_convergen['Fitness_con'].values[0]} sehingga mendapatkan akurasi sebesar {akurasi*100:.2f}%")

    with tab2 :
        option = st.selectbox('METODE',('SVM', 'GA-SVM'))
        document = st.text_input('masukan kalimat',value="Saran aja buat {name_app} kl bisa limit nya jgn di batasi per 30hari,3bulan,6bulan,12bulan.. Jd kl mau blnj agak susah..kadang linit 30hari g mencukupi.apalagi yg 12 bulan udah pake sekali g bisa pake lagi,kudu lunas dl.🤣🤣🤭🤭 Saran aja,kyk di sebelah limit global sekian juta sy malah suka pake yg dr Aku*******sering sy pake.mau blnj tgl pilih tenor nya. Dari 1bulan,3bulan,6bulan,12bulan.")
        kcleansing = cleansing(document)
        kcasefolding = casefolding(kcleansing)
        ktokenizing = tokenizing(kcasefolding)
        kstemming = stemming(ktokenizing)
        knegasi= handle_negation(kstemming)
        kslangword = slangword(knegasi)
        kstopword = stopword(kslangword)
        kdatastr = str(kstopword)
        ktfidf =vectorizer.transform([kdatastr])
        if (option == 'SVM') :
            if st.button('klasifikasi') :
                st.write('Hasil pengujian dengan metode',option)
                # Making the SVM Classifer
                svmbiasa = model_svm(None,None,x_train,y_train)
                
                predictions = svmbiasa.predict(ktfidf)
                st.write('hasil cleansing :',str(kcleansing))
                st.write('hasil casefolding :',str(kcasefolding))
                st.write('hasil tokenizing :',str(ktokenizing))
                st.write('hasil stemming :',str(kstemming))
                st.write('hasil negasi :',str(knegasi))
                st.write('hasil word normalization :',str(kslangword))
                st.write('hasil stopword :',str(kstopword))
                if not kstemming:
                    st.write("Maaf mohon inputkan kalimat lagi :)")
                elif predictions == 2:
                    st.write(f"karena nilai prediksi adalah 1 maka termasuk kelas Sentimen Positif")
                elif predictions == 0:
                    st.write(f"karena nilai prediksi adalah -1 maka termasuk kelas Sentimen Negatif")
                elif predictions == 1:
                    st.write(f"karena nilai prediksi adalah 0 maka termasuk kelas Sentimen netral")
            else:
                st.write('hasil akan tampil disini :)') 
        elif (option == 'GA-SVM') :
            
            bestCbaru = st.session_state["bestCbaru"]
            bestGammabaru = st.session_state["bestGammabaru"]

            if st.button('klasifikasi') :
                st.write(f'Hasil pengujian dengan metode',option,f'parameter yang digunakan C={bestCbaru:.2f} dan gamma={bestGammabaru:.2f}')
                svmGAbaru = model_svm(bestCbaru,bestGammabaru,x_train,y_train)
                # Making the SVM Classifer
                predictions = svmGAbaru.predict(ktfidf)
                st.write('hasil cleansing :',str(kcleansing))
                st.write('hasil casefolding :',str(kcasefolding))
                st.write('hasil tokenizing :',str(ktokenizing))
                st.write('hasil stopword :',str(kstopword))
                st.write('hasil word normalization :',str(kslangword))
                st.write('hasil stemming :',str(kstemming))
                if not kstemming:
                    st.write("Maaf mohon inputkan kalimat lagi :)")
                elif predictions == 2:
                    st.write(f"karena nilai prediksi adalah 1 maka termasuk kelas Sentimen Positif")
                elif predictions == 0:
                    st.write(f"karena nilai prediksi adalah -1 maka termasuk kelas Sentimen Negatif")
                elif predictions == 1:
                    st.write(f"karena nilai prediksi adalah 0 maka termasuk kelas Sentimen netral")
            else:
                st.write('hasil akan tampil disini :)')  
            

elif(selected == 'Report') :
    st.header('Halaman Report')
    st.write('evaluasi model menggunakan confusion matrix')
    
    GAbaru=pd.read_csv('hasil_optimasi_GA.csv')
    bestCbaru = GAbaru['C_best'].iloc[-1]
    bestGammabaru = GAbaru['Gamma_best'].iloc[-1]
    st.subheader('hasil evaluasi metode svm')
    svmbiasa = model_svm(None,None,x_train,y_train)
    class_report(svmbiasa,x_test,y_test)
    st.subheader('hasil evaluasi metode gasvm')
    svmGAbaru = model_svm(bestCbaru,bestGammabaru,x_train,y_train)
    class_report(svmGAbaru,x_test,y_test)
