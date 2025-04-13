from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def output_tfidf(dataset,column_name):
    vectorizer = TfidfVectorizer()
    # Transformasi data hasil prepro menggunakan TF-IDF
    tfidf = vectorizer.fit_transform(dataset[f'{column_name}'])
    
    # Mendapatkan daftar kata yang digunakan dalam TF-IDF
    feature_names = vectorizer.get_feature_names_out()

    # Membuat DataFrame kosong untuk menyimpan nilai TF-IDF
    tfidf_df = pd.DataFrame(columns=['TF-IDF'])

    # Mengisi DataFrame dengan nilai TF-IDF yang tidak nol
    for i, doc in enumerate(dataset[f'{column_name}']):
        doc_tfidf = tfidf[i]
        non_zero_indices = doc_tfidf.nonzero()[1]
        tfidf_values = doc_tfidf[0, non_zero_indices].toarray()[0]
        tfidf_dict = {feature_names[idx]: tfidf_values[j] for j, idx in enumerate(non_zero_indices)}
        tfidf_df.loc[i] = [' '.join(f'({feature_name}, {tfidf_dict[feature_name]:.3f})' for feature_name in tfidf_dict)]
    # Convert the results dictionary to a Pandas dataframe
    dataset = pd.concat([dataset, tfidf_df], axis=1)

    return tfidf,dataset,vectorizer