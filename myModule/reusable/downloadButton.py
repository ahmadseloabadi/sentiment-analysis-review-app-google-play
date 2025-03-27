import streamlit as st
@st.cache_data
def convert_for_download(df):
    return df.to_csv(index=False).encode("utf-8")

def download_data(data,name):
    csv = convert_for_download(data)
    st.download_button(
        label=f"Download hasil {name}",
        data=csv,
        file_name=f"data hasil {name}.csv",
        mime="text/csv",
        icon=":material/download:",
    )