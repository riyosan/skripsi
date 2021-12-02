import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser

def upload_dataset():
	with st.sidebar.header("1. Upload your CSV data"):
		data=st.file_uploader("Upload your input CSV file",type=['csv'])
		return data
def prepro(df):
	#parsing
	df.first_open=[parser.parse(i) for i in df.first_open]
	df.enrolled_date=[parser.parse(i) if isinstance(i, str)else i for i in df.enrolled_date]
#mengurangi enrolled_date dengan firs_open
	df['selisih']=(df.enrolled_date-df.first_open).astype('timedelta64[h]')
	df.loc[df.selisih>24, 'enrolled'] = 0
#import top_screen
	top_screens=pd.read_csv('/content/skripsi/top_screens.csv')
	top_screens=np.array(top_screens.loc[:,'top_screens'])
	for i in top_screens:
		df[i]=df.screen_list.str.contains(i).astype(int)
	for i in top_screens:
		df['screen_list']=df.screen_list.str.replace(i+',','')
#menghapus double layar
	layar_loan = ['Loan','Loan2','Loan3','Loan4']
	df['jumlah_loan']=df[layar_loan].sum(axis=1)
	df.drop(columns=layar_loan, inplace=True)

	layar_saving = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
	df['jumlah_loan']=df[layar_saving].sum(axis=1)
	df.drop(columns=layar_saving, inplace=True)

	layar_credit = ['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
	df['jumlah_credit']=df[layar_credit].sum(axis=1)
	df.drop(columns=layar_credit, inplace=True)

	layar_cc = ['CC1','CC1Category','CC3']
	df['jumlah_cc']=df[layar_cc].sum(axis=1)
	df.drop(columns=layar_cc, inplace=True)
	df['lainnya']=df.screen_list.str.count(',')
	st.write(df.head(10))
#menghitung jumlah screen
	df['screen_list'] = df.screen_list.astype(str) + ','
	df['num_screens'] = df.screen_list.str.count(',')
	df.drop(columns=['numscreens'], inplace=True)
#mengubah kolom hour
	df.hour=df.hour.str.slice(1,3).astype(int)

#mendefenisikan variabel numerik
	df_numerik=df.drop(columns=['user','first_open','screen_list','enrolled_date','selisih'], inplace=False)
	df_types = df.dtypes.astype(str)
	st.write(df_numerik.head(10))
	df_types = df.dtypes.astype(str)
	st.write(df_types)
#membuat korelasi matrik
	korelasi = df_numerik.drop(columns=['enrolled'], inplace=False).corrwith(df_numerik.enrolled)
	plot=korelasi.plot.bar(title='korelasi variabel')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	df_numerik.to_csv('/content/skripsi/data', index=False)
def app():
	global data
	filenya=upload_dataset()
	if filenya is not None:
		df=pd.read_csv(filenya)
		if st.button("Load Data"):
			st.write(df)
			prepro(df)
	else:
		if st.button('Press to use Example Dataset'):
			df = pd.read_csv('/content/skripsi/fintech_data.csv')
			prepro(df)
