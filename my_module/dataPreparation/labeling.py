import streamlit as st
import pandas as pd


def manual_labeling(dataset_path):
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

