import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser
import joblib

def upload_dataset_pred():
  with st.header("1. Upload your CSV data"):
    data_pred=st.file_uploader("Upload your input CSV file",type=['csv'])
    return data_pred
def prepro_pred(pred):
  #menghitung jumlah screen
  pred['screen_list'] = pred.screen_list.astype(str) + ','
  pred['num_screens'] = pred.screen_list.str.count(',')
  #menghapus kolom numsreens yng lama
  pred.drop(columns=['numscreens'], inplace=True)
  #mengubah kolom hour
  pred.hour=pred.hour.str.slice(1,3).astype(int)
  #karena tipe data first_open dan enrolled_date itu adalah string, maka perlu diubah ke datetime
  pred.first_open=[parser.parse(i) for i in pred.first_open]
  #import top_screen
  top_screens=pd.read_csv('/content/skripsi/top_screens.csv')
  top_screens=np.array(top_screens.loc[:,'top_screens'])
  pred_copy = pred.copy()
  for i in top_screens:
      pred[i]=pred.screen_list.str.contains(i).astype(int)
  for i in top_screens:
      pred['screen_list']=pred.screen_list.str.replace(i+',','')
  #menghitung jumlah item non top screen yang(tersisa) ada di screenlist
  pred['lainnya']=pred.screen_list.str.count(',')
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
  #mendefenisikan variabel numerik
  pred_numerik=pred.drop(columns=['first_open','screen_list'], inplace=False)
  st.write(pred_numerik)
  scaler = joblib.load('/content/skripsi/data/minmax_scaler.joblib')
  kolom_training = [i for i in pred_numerik.columns if i not in['user']]
  pred_numerik[kolom_training] = scaler.transform(pred_numerik[kolom_training])
  fitur = pd.read_csv('/content/skripsi/data/fitur_pilihan.csv')
  fitur = fitur['0'].tolist()
  pred_numerik = pred_numerik[fitur]
  model = joblib.load('/content/skripsi/data/stack_model.pkl')
  prediksi = model.predict(pred_numerik)
  st.write(real_prediksi)
def app():
  global data_pred
  filenya=upload_dataset_pred()
  if filenya is not None:
    pred=pd.read_csv(filenya)
    if st.button("predict Data"):
      st.write(pred)
      prepro_pred(pred)
  else:
    if st.button('Press to use Example Dataset'):
      pred = pd.read_csv('/content/skripsi/testing.csv')
      prepro_pred(pred)
