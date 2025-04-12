import streamlit as st
import pickle
@st.cache_data
def convert_for_download(df):
    return df.to_csv(index=False).encode("utf-8")

def download_data(data,proses,name_app):
    csv = convert_for_download(data)
    st.download_button(
        label=f"Download hasil {proses}",
        data=csv,
        file_name=f"Download hasil {proses} ulasan aplikasi {name_app}.csv",
        mime="text/csv",
        icon=":material/download:",
    )

def download_model(model,name_model):
    st.download_button(
    "Download Model",
    data=pickle.dumps(model),
    file_name=f"{name_model}.pkl",
    icon=":material/download:",
    )   