import streamlit as st
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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