import pickle
import streamlit as st
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB


def load_model(uploaded_file):
    try:
        model = pickle.load(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None
    
def knn_model(x_train, y_train,x_test, y_test):

    # Mengaktifkan fungsi klasifikasi
    klasifikasi = KNeighborsClassifier()
    # Memasukkan data training pada fungsi klasifikasi
    klasifikasi.fit(x_train, y_train)
    # Menentukan hasil prediksi dari x_test
    prediction_test = klasifikasi.predict(x_test)
    # Menentukan probabilitas hasil prediksi
    klasifikasi.predict_proba(x_test)
    st.text_area(classification_report(y_test, prediction_test))

def complement_NB_model(x_train, y_train,x_test, y_test):

    # Creating and training the Complement Naive Bayes Classifier
    classifier = ComplementNB()
    classifier.fit(x_train, y_train)

    # Evaluating the classifier
    prediction_test = classifier.predict(x_test)
    prediction_train = classifier.predict(x_train)
    st.text_area(classification_report(y_test, prediction_test))
def multinomial_NB_model(x_train, y_train,x_test, y_test):

    # Creating and training the Complement Naive Bayes Classifier
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)

    # Evaluating the classifier
    prediction_test = classifier.predict(x_test)
    prediction_train = classifier.predict(x_train)
    st.text_area(classification_report(y_test, prediction_test))

def svm_model(x_train, y_train,x_test, y_test):
    # Making the SVM Classifer
    classifier = svm.SVC()

    # Training the model on the training data and labels
    classifier.fit(x_train, y_train)

    # Using the model to predict the labels of the test data
    prediction_test = classifier.predict(x_test)
    # Printing the results
    st.text_area(classification_report(y_test, prediction_test))


