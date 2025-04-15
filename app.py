#import library
import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import re


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer

from sklearn.preprocessing import LabelEncoder
from sklearn import svm


import myModule.dataPreparation.preprocessing as prepro
import myModule.dataPreparation.labeling as labeling
import myModule.dataPreparation.tfidf as tfidf
import myModule.dataPreparation.splitdata as splitdata

from myModule.dataGathering.scraping import scrapping_play_store
from myModule.reusable.downloadButton import download_data,download_model
from myModule.dataPreparation.data_visual import output_dataset,report_dataset_final

# Set page layout and title
st.set_page_config(page_title="review google play app")

vectorizer = TfidfVectorizer()
Encoder = LabelEncoder()


def model_svm(C,gamma,x_train,y_train):
    if C is not None:
        modelsvm = svm.SVC(C=C,gamma=gamma)
    else:
        modelsvm = svm.SVC()
    modelsvm.fit(x_train, y_train)
    return modelsvm



kfold=5




#side bar
with st.sidebar :
    selected = option_menu('sentimen analisis',['Home','Data Preparation','Modeling'])

if(selected == 'Home') :
    tab1,tab2=st.tabs(['Main','Scrapping'])
    with tab1 :
        st.title(' ULASAN APLIKASI PADA GOOGLE PLAY STORE ')

    with tab2:
        st.write(f'scrapping data ulasan aplikasi pada google play store')
        st.info("https://play.google.com/store/apps/details?id=com.reddit.frontpage&hl=id")
        url_app=st.text_input("masukan link aplikasi google playstore :")
        if st.button('start scrapping') :
            scrapping_result=scrapping_play_store(url_app)
            st.session_state.scarping_data=scrapping_result
        
        if "scarping_data" in st.session_state:
            name_app=st.session_state.name_app
            scrapping_result=st.session_state.scarping_data
            st.dataframe(scrapping_result)
            df = scrapping_result[:100]
            download_data(df,"scraping",name_app)
            

elif(selected == 'Data Preparation') :
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['Text preprosesing','TF-IDF','Labeling','SMOTE','Split Data','Dataset Overview'])

    with tab1 :

        file_prepro = st.file_uploader("masukan data yang akan dilakukan preprocessing", key="preprocewssing", type='csv')

        if file_prepro is not None:
            dataset = pd.read_csv(file_prepro)
            st.write('proses untuk menampilkan hasil text preprocessing dan tf-idf mungkin membutuhkan waktu yang cukup lama')
            kolom = st.selectbox('masukan nama kolom ulasan/review pada data yang di input',dataset.columns,key="preprocessing")
            st.write('tampilan dataset yang di input')
            st.write(dataset)

            if st.button('start text preprocessing') :
                dataset['Cleansing']= dataset[kolom].apply(prepro.cleansing)
                dataset['CaseFolding']= dataset['Cleansing'].apply(prepro.casefolding)
                dataset['Tokenizing']= dataset['CaseFolding'].apply(prepro.tokenizing)
                dataset['Stemming']= dataset['Tokenizing'].apply(prepro.stemming)
                dataset['Negasi']= dataset['Stemming'].apply(prepro.handle_negation)
                dataset['Word Normalization']= dataset['Negasi'].apply(prepro.slangword)
                dataset['stopword']= dataset['Word Normalization'].apply(prepro.stopword)
                dataset['Stopword Removal'] = dataset['stopword'].apply(' '.join)
                dataset.drop(columns='stopword',inplace=True)
                prepro=True
                st.dataframe(dataset,use_container_width=True)
                st.session_state.dataset = dataset
                st.session_state.prepro = prepro

        if "dataset" in st.session_state and "prepro" in st.session_state:
            dataset=st.session_state.dataset
            st.write('tampilan hasil cleansing')
            cleansing = dataset[[kolom,'Cleansing']]
            st.dataframe(cleansing)

            st.write('tampilan hasil casefolding')
            casefolding= dataset[['Cleansing','CaseFolding']]
            st.dataframe(casefolding)

            st.write('tampilan hasil tokenizing')
            tokenizing= dataset[['CaseFolding','Tokenizing']]
            st.dataframe(tokenizing)

            st.write('tampilan hasil stemming')
            stemming= dataset[['Tokenizing','Stemming']]
            st.dataframe(stemming)

            st.write('tampilan hasil Negasi')
            negasi= dataset[['Stemming','Negasi']]
            st.dataframe(negasi)

            st.write('tampilan hasil word normalization')
            wordnormalization= dataset[['Negasi','Word Normalization']]
            st.dataframe(wordnormalization)

            st.write('tampilan hasil stopword')
            stopword= dataset[['Word Normalization','Stopword Removal']]
            st.dataframe(stopword)
            
            file_name = file_prepro.name
            pattern = r"Download hasil .+ ulasan aplikasi ([\w\s]+)\.csv"  # Pola untuk mengambil kata terakhir sebelum .csv
            match = re.search(pattern, file_name)
            
            name_app_from_file = match.group(1)
            download_data(dataset,"preprocessing",name_app_from_file)

    with tab2:
        uploaded_file = st.file_uploader("masukan data yang akan dilakukan pembobotan TF-IDF", key="TF-IDF", type='csv')

        if uploaded_file is not None:
            
            dataset = pd.read_csv(uploaded_file)
            st.write('proses untuk menampilkan hasil text preprocessing dan tf-idf mungkin membutuhkan waktu yang cukup lama')
            kolom = st.selectbox('masukan nama kolom stopword pada data yang diinput',dataset.columns,key="tfidf")
            st.write('tampilan dataset yang di input')
            st.write(dataset)

            if st.button('Start TF-IDF'):
                with st.spinner("Sedang melakukan pembobotan TF-IDF"):
                    tfidf_model, df_tfidf ,vectorizer = tfidf.output_tfidf(dataset, kolom)
                    st.session_state.tfidf_model = tfidf_model
                    st.session_state.df_tfidf = df_tfidf
                    st.session_state.vectorizer_model = vectorizer

                    st.toast('Berhasil melakukan pembobotan TF-IDF')

        # Jika hasil sudah ada di session_state
        if "df_tfidf" in st.session_state and "tfidf_model" in st.session_state and "vectorizer_model" in st.session_state:
            df_tfidf = st.session_state.df_tfidf
            tfidf_model = st.session_state.tfidf_model
            vectorizer=st.session_state.vectorizer_model

            file_name = uploaded_file.name
            pattern = r"Download hasil .+ ulasan aplikasi ([\w\s]+)\.csv"  # Pola untuk mengambil kata terakhir sebelum .csv
            match = re.search(pattern, file_name)
            
            name_app = match.group(1)

            st.write('Tampilan hasil pembobotan TF-IDF')
            st.dataframe(df_tfidf[['content','Stopword Removal','TF-IDF']], use_container_width=True)

            download_data(df_tfidf, "pembobotan TF-IDF", name_app)
            download_model(tfidf_model, "TF-IDF")
            download_model(vectorizer,"vectorizer")

    with tab3 :
        labeling_data = st.file_uploader("masukan data yang akan dilabeli", key="labeling_data", type='csv')
        if labeling_data is not None:
            dataset = pd.read_csv(labeling_data)
            file_name = labeling_data.name
            pattern = r"Download hasil .+ ulasan aplikasi ([\w\s]+)\.csv"  # Pola untuk mengambil kata terakhir sebelum .csv
            match = re.search(pattern, file_name)
            extracted_name = match.group(1)
            option_label=st.selectbox('milih metode pelabelan :',('manual',"vader","textblob","inset_lexicon"),key="pelabelan")

            if option_label == 'manual':
                labeling.manual_labeling(dataset,extracted_name)
            if option_label == 'vader':
                if st.button('Start labeling data'):
                    with st.spinner("Sedang melakukan pelabelan data..."):
                        data=labeling.vader_labeling(dataset)
            if option_label == 'textblob':
                if st.button('Start labeling data'):
                    with st.spinner("Sedang melakukan pelabelan data..."):
                        data=labeling.textblob_labeling(dataset)
            if option_label == 'inset_lexicon':
                if st.button('Start labeling data'):
                    with st.spinner("Sedang melakukan pelabelan data..."):
                        data=labeling.inset_labeling(dataset)
            

                
    with tab4:
        st.write('tempat smote')
    with tab5:
        st.write("ini tempat split dataset")
    with tab6 :
        file_final = st.file_uploader("masukan data final", key="datasettt_final", type='csv')

        if file_final is not None:
            file = pd.read_csv(file_final)
            file_name = file_final.name
            pattern = r"Download hasil .+ ulasan aplikasi ([\w\s]+)\.csv"  # Pola untuk mengambil kata terakhir sebelum .csv
            match = re.search(pattern, file_name)
            extracted_name = match.group(1)
            st.dataframe(file)
            ulasan = st.selectbox('masukan nama kolom ulasan data',file.columns,key="data overview")
            label = st.selectbox('masukan nama kolom labeling data',file.columns,key="data overview label")
            if st.button("overview dataset"):
                kolom_ulasan =file[ulasan]
                kolom_label =file[label]
                report_dataset_final(file,kolom_ulasan,kolom_label,extracted_name)
elif(selected == 'Modeling') :
    tab1,tab2,tab3=st.tabs(['Model','Testing','Evaluation'])
    with tab1:
        st.write('training model')
    with tab2 :
        option = st.selectbox('METODE',('SVM', 'GA-SVM'))
        document = st.text_input('masukan kalimat',value="Saran aja buat {name_app} kl bisa limit nya jgn di batasi per 30hari,3bulan,6bulan,12bulan.. Jd kl mau blnj agak susah..kadang linit 30hari g mencukupi.apalagi yg 12 bulan udah pake sekali g bisa pake lagi,kudu lunas dl.🤣🤣🤭🤭 Saran aja,kyk di sebelah limit global sekian juta sy malah suka pake yg dr Aku*******sering sy pake.mau blnj tgl pilih tenor nya. Dari 1bulan,3bulan,6bulan,12bulan.")
        kcleansing = prepro.cleansing(document)
        kcasefolding = prepro.casefolding(kcleansing)
        ktokenizing = prepro.tokenizing(kcasefolding)
        kstemming = prepro.stemming(ktokenizing)
        knegasi= prepro.handle_negation(kstemming)
        kslangword = prepro.slangword(knegasi)
        kstopword = prepro.stopword(kslangword)
        kdatastr = str(kstopword)
        ktfidf =vectorizer.transform([kdatastr])
        if (option == 'SVM') :
            if st.button('klasifikasi') :
                st.write('Hasil pengujian dengan metode',option)
                # Making the SVM Classifer
                # svmbiasa = model_svm(None,None,x_train,y_train)
                
                # predictions = svmbiasa.predict(ktfidf)
                st.write('hasil cleansing :',str(kcleansing))
                st.write('hasil casefolding :',str(kcasefolding))
                st.write('hasil tokenizing :',str(ktokenizing))
                st.write('hasil stemming :',str(kstemming))
                st.write('hasil negasi :',str(knegasi))
                st.write('hasil word normalization :',str(kslangword))
                st.write('hasil stopword :',str(kstopword))
                # if not kstemming:
                #     st.write("Maaf mohon inputkan kalimat lagi :)")
                # elif predictions == 2:
                #     st.write(f"karena nilai prediksi adalah 1 maka termasuk kelas Sentimen Positif")
                # elif predictions == 0:
                #     st.write(f"karena nilai prediksi adalah -1 maka termasuk kelas Sentimen Negatif")
                # elif predictions == 1:
                #     st.write(f"karena nilai prediksi adalah 0 maka termasuk kelas Sentimen netral")
            else:
                st.write('hasil akan tampil disini :)') 
        elif (option == 'GA-SVM') :
            
            bestCbaru = st.session_state["bestCbaru"]
            bestGammabaru = st.session_state["bestGammabaru"]

            if st.button('klasifikasi') :
                st.write(f'Hasil pengujian dengan metode',option,f'parameter yang digunakan C={bestCbaru:.2f} dan gamma={bestGammabaru:.2f}')
                # svmGAbaru = model_svm(bestCbaru,bestGammabaru,x_train,y_train)
                # Making the SVM Classifer
                # predictions = svmGAbaru.predict(ktfidf)
                st.write('hasil cleansing :',str(kcleansing))
                st.write('hasil casefolding :',str(kcasefolding))
                st.write('hasil tokenizing :',str(ktokenizing))
                st.write('hasil stopword :',str(kstopword))
                st.write('hasil word normalization :',str(kslangword))
                st.write('hasil stemming :',str(kstemming))
                # if not kstemming:
                #     st.write("Maaf mohon inputkan kalimat lagi :)")
                # elif predictions == 2:
                #     st.write(f"karena nilai prediksi adalah 1 maka termasuk kelas Sentimen Positif")
                # elif predictions == 0:
                #     st.write(f"karena nilai prediksi adalah -1 maka termasuk kelas Sentimen Negatif")
                # elif predictions == 1:
                #     st.write(f"karena nilai prediksi adalah 0 maka termasuk kelas Sentimen netral")
            else:
                st.write('hasil akan tampil disini :)')  
            
    with tab3 :
        st.header('Halaman Report')
        st.write('evaluasi model menggunakan confusion matrix')
        
        GAbaru=pd.read_csv('hasil_optimasi_GA.csv')
        bestCbaru = GAbaru['C_best'].iloc[-1]
        bestGammabaru = GAbaru['Gamma_best'].iloc[-1]
        st.subheader('hasil evaluasi metode svm')
        # svmbiasa = model_svm(None,None,x_train,y_train)
        # class_report(svmbiasa,x_test,y_test)
        st.subheader('hasil evaluasi metode gasvm')
        # svmGAbaru = model_svm(bestCbaru,bestGammabaru,x_train,y_train)
        # class_report(svmGAbaru,x_test,y_test)
