import pandas as pd
import re
import numpy as np
import streamlit as st
from google_play_scraper import Sort, reviews_all, app
from google_play_scraper.exceptions import NotFoundError

def extract_app_details(url):
    """Ekstrak app ID, bahasa, dan negara dari URL Play Store."""
    app_id_match = re.search(r'id=([^&]+)', url)
    lang_match = re.search(r'hl=([^&]+)', url)
    country_match = re.search(r'gl=([^&]+)', url)
    
    app_id = app_id_match.group(1) if app_id_match else None
    lang = lang_match.group(1) if lang_match else 'id'  # Default ke Inggris jika tidak ada
    country = country_match.group(1) if country_match else 'ID'  # Default ke US jika tidak ada
    
    return app_id, lang, country

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
        st.session_state['is_scrap'] = True
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
        
        st.write("jika data sudah sesuai silahkan download data untuk proses selanjutnya :)")
        csv = sorted_df.to_csv().encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ulasan aplikasi {name_app}.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    finally:
        if 'is_scrap' not in st.session_state :
            st.toast("silahkan lakukan scraping data terlebih dahulu")
