import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser

def upload_dataset_pred():
    with st.header("1. Upload your CSV data"):
        data_pred=st.file_uploader("Upload your input CSV file",type=['csv'])
        return data_pred
def prepro_pred(pred):
#import top_screen
    top_screens=pd.read_csv('/content/skripsi/top_screens.csv')
    top_screens=np.array(top_screens.loc[:,'top_screens'])
    for i in top_screens:
        pred[i]=pred.screen_list.str.contains(i).astype(int)
    for i in top_screens:
        pred['screen_list']=pred.screen_list.str.replace(i+',','')
#menghapus double layar
    layar_loan = ['Loan','Loan2','Loan3','Loan4']
    pred['jumlah_loan']=pred[layar_loan].sum(axis=1)
    pred.drop(columns=layar_loan, inplace=True)

    layar_saving = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
    pred['jumlah_loan']=pred[layar_saving].sum(axis=1)
    pred.drop(columns=layar_saving, inplace=True)

    layar_credit = ['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
    pred['jumlah_credit']=pred[layar_credit].sum(axis=1)
    pred.drop(columns=layar_credit, inplace=True)

    layar_cc = ['CC1','CC1Category','CC3']
    pred['jumlah_cc']=pred[layar_cc].sum(axis=1)
    pred.drop(columns=layar_cc, inplace=True)
    pred['lainnya']=pred.screen_list.str.count(',')
    st.write(pred.head(10))
#menghitung jumlah screen
    pred['screen_list'] = pred.screen_list.astype(str) + ','
    pred['num_screens'] = pred.screen_list.str.count(',')
    pred.drop(columns=['numscreens'], inplace=True)
#mengubah kolom hour
    pred.hour=pred.hour.str.slice(1,3).astype(int)

#mendefenisikan variabel numerik
    pred_numerik=pred.drop(columns=['first_open','screen_list','enrolled_date','selisih'], inplace=False)
    pred_numerik=pred.drop(columns=['user'], inplace=False)
    pred_types = pred.dtypes.astype(str)
    st.write(pred_numerik.head(10))
    pred_types = pred.dtypes.astype(str)
    st.write(pred_types)
def app():
    global data_pred
    filenya=upload_dataset()
    if filenya is not None:
        pred=pd.read_csv(filenya)
        if st.button("Load Data"):
            st.write(pred)
            prepro_pred(pred)
    else:
        if st.button('Press to use Example Dataset'):
            pred = pd.read_csv('/content/skripsi/pred_copy.csv')
            prepro_pred(pred)
