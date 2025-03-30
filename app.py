#import library
import streamlit as st
from streamlit_option_menu import option_menu


import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import re
import numpy as np
import random as rd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

import myModule.ga.Genetic_Algorithm as svm_hp_opt
import myModule.dataPreparation.preprocessing as prepro
import myModule.dataPreparation.labeling as labeling
from myModule.dataGathering.scraping import scrapping_play_store
from myModule.reusable.downloadButton import download_data
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
    selected = option_menu('sentimen analisis',['Home','Data Preparation','Modeling'])

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
        st.info("https://play.google.com/store/apps/details?id=com.reddit.frontpage&hl=id")
        url_app=st.text_input("masukan link aplikasi google playstore :")
        if st.button('start scrapping') :
            scrapping_result=scrapping_play_store(url_app)
            

elif(selected == 'Data Preparation') :
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['Text preprosesing','TF-IDF','Labeling','Dataset','SMOTE','Overview'])

    with tab1 :

        uploaded_file = st.file_uploader("masukan data yang akan dilakukan preprocessing", key="preprocewssing", type='csv')

        if uploaded_file is not None:
            
            dataset = pd.read_csv(uploaded_file)
            st.write('proses untuk menampilkan hasil text preprocessing dan tf-idf mungkin membutuhkan waktu yang cukup lama')
            kolom = st.text_input('masukan nama kolom ulasan/review pada data yang di input',value="content")
            st.write('tampilan dataset yang di input')
            st.write(dataset)

            if st.button('start text preprocessing') :
                dataset['Cleansing']= dataset[kolom].apply(prepro.cleansing)
                st.write('tampilan hasil cleansing')
                cleansing = dataset[[kolom,'Cleansing']]
                st.dataframe(cleansing)

                dataset['CaseFolding']= dataset['Cleansing'].apply(prepro.casefolding)
                st.write('tampilan hasil casefolding')
                casefolding= dataset[['Cleansing','CaseFolding']]
                st.dataframe(casefolding)
                
                dataset['Tokenizing']= dataset['CaseFolding'].apply(prepro.tokenizing)
                st.write('tampilan hasil tokenizing')
                tokenizing= dataset[['CaseFolding','Tokenizing']]
                st.dataframe(tokenizing)
                
                dataset['Stemming']= dataset['Tokenizing'].apply(prepro.stemming)
                st.write('tampilan hasil stemming')
                stemming= dataset[['Tokenizing','Stemming']]
                st.dataframe(stemming)

                dataset['Negasi']= dataset['Stemming'].apply(prepro.handle_negation)
                st.write('tampilan hasil Negasi')
                negasi= dataset[['Stemming','Negasi']]
                st.dataframe(negasi)

                dataset['Word Normalization']= dataset['Negasi'].apply(prepro.slangword)
                st.write('tampilan hasil word normalization')
                wordnormalization= dataset[['negasi','Word Normalization']]
                st.dataframe(wordnormalization)

                dataset['stopword']= dataset['Word Normalization'].apply(prepro.stopword)
                st.write('tampilan hasil stopword')
                stopword= dataset[['Word Normalization','stopword']]
                #merubah list ke str 
                dataset['Stopword Removal'] = dataset['stopword'].apply(' '.join)
                dataset.drop(columns='stopword',inplace=True)
                st.dataframe(stopword)
                prepro=True
                st.dataframe(dataset,use_container_width=True)
                if 'name_app' not in st.session_state:
                    file_name = uploaded_file.name
                    pattern = r"data hasil(?: .+)? (\w+)\.csv"  # Pola untuk mengambil kata terakhir sebelum .csv
                    match = re.search(pattern, file_name)
                    name_app_from_file = match.group(1)
                    download_data(dataset,"preprocessing",name_app_from_file)
                else:
                    name_app=st.session_state['name_app']
                    download_data(dataset,"preprocessing",name_app)

    with tab2:
        uploaded_file = st.file_uploader("masukan data yang akan dilakukan pembobotan TF-IDF", key="TF-IDF", type='csv')

        if uploaded_file is not None:
            
            dataset = pd.read_csv(uploaded_file)
            st.write('proses untuk menampilkan hasil text preprocessing dan tf-idf mungkin membutuhkan waktu yang cukup lama')
            kolom = st.text_input('masukan nama kolom stopword pada data yang diinput',value="Stopword Removal")
            st.write('tampilan dataset yang di input')
            st.write(dataset)

            if st.button('start TF-IDF') :
                with st.spinner("Sedang melakukan pembobotan TF-IDF"):
                    df_tfidf=prepro.output_tfidf(dataset,kolom)
                st.toast('berhasil melakukan pembobotan TF-IDF')

                st.write('tampilan hasil pembobotan TF-IDF')

                st.dataframe(df_tfidf[['content','Stopword Removal','TF-IDF']],use_container_width=True)
                download_data(df_tfidf,"pembobotan TF-IDF")
    
    with tab3 :
        file_labeling = st.file_uploader("masukan data yang akan dilabeli", key="labeling_data", type='csv')
        if file_labeling is not None:
            dataset = pd.read_csv(file_labeling)
            option_label=st.selectbox('milih metode pelabelan :',('manual',"vader","textblob","inset_lexicon"),key="pelabelan")

            if option_label == 'manual':
                labeling.manual_labeling(dataset)
            if option_label == 'vader':
                with st.spinner("Sedang melakukan pelabelan data..."):
                    labeling.vader_labeling(dataset)
                
            if option_label == 'textblob':
                labeling.textblob_labeling(dataset)
            if option_label == 'inset_lexicon':
                labeling.inset_labeling(dataset)
            
    
    with tab4 :        
        file_labeling = st.file_uploader("masukan data yg sudah dilabeli", key="datasettt", type='csv')

        if file_labeling is not None:
            file = pd.read_csv(file_labeling)
            file_name = file_labeling.name
            pattern = r"data hasil(?: .+)? (\w+)\.csv"  # Pola untuk mengambil kata terakhir sebelum .csv
            match = re.search(pattern, file_name)
            extracted_name = match.group(1)
            st.dataframe(file)
            ulasan = st.text_input('masukan nama kolom ulasan data',value="content")
            label = st.text_input('masukan nama kolom labeling data',value="sentimen")
            if st.button("tampilkan dataset"):
                kolom_ulasan =file[ulasan]
                kolom_label =file[label]

                output_dataset(file,extracted_name)
    with tab5:
        st.write('tempat smote')
    with tab6 :
        file_final = st.file_uploader("masukan data final", key="datasettt_final", type='csv')

        if file_final is not None:
            file = pd.read_csv(file_final)
            file_name = file_final.name
            pattern = r"data hasil(?: .+)? (\w+)\.csv"  # Pola untuk mengambil kata terakhir sebelum .csv
            match = re.search(pattern, file_name)
            extracted_name = match.group(1)
            ulasan = st.text_input('masukan nama kolom ulasan data',value="content")
            label = st.text_input('masukan nama kolom labeling data',value="sentimen")
            
            kolom_ulasan =file[ulasan]
            kolom_label =file[label]
            report_dataset_final(file,kolom_ulasan,kolom_label,extracted_name)
elif(selected == 'Modeling') :
    tab1,tab2,tab3=st.tabs(['Model','Evaluation','Testing'])
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
            
    with tab3 :
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
