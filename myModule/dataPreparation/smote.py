from imblearn.over_sampling import SMOTE

def smote(tfidf,dataset):
    x_res =vectorizer.fit_transform(ulasan['Stemming'])
    y_res =ulasan['Sentimen']
    smote = SMOTE(sampling_strategy='auto')
    X_smote, Y_smote = smote.fit_resample(x_res, y_res)
    df = pd.DataFrame(X_smote)
    df.rename(columns={0:'term'}, inplace=True)
    df['sentimen'] = Y_smote
    # mengembalikan kalimat asli dari tfidf
    feature_names = vectorizer.get_feature_names_out()

    kalimat_asli = []
    for index, row in df.iterrows():
        vektor_ulasan = X_smote[index]
        kata_kunci = [feature_names[i] for i in vektor_ulasan.indices]
        kalimat_asli.append(' '.join(kata_kunci))

    # tambahkan kolom baru dengan kalimat asli ke dalam data frame
    df['kalimat_asli'] = kalimat_asli
    df.to_csv('data_smote.csv', index=False)
    #mengambil data sintetik
    df_sintetik = df.iloc[1000:]
    #menyimpan dalam bentuk csv
    df_sintetik.to_csv('data_sintetik.csv', index=False)
    df