import streamlit as st
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from apps import home
import os
from dateutil import parser
def app():
	if 'main_data.csv' not in os.listdir('data'):
		st.markdown("Please upload data through `Home` page!")
	else:
		df = pd.read_csv('data/main_data.csv')
		var_enrolled = np.array(df['enrolled'])
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(np.array(df.drop(labels=['enrolled'], axis=1)), var_enrolled, test_size=0.2, random_state=111)
		st.write(X_test)
	# from sklearn.feature_selection import mutual_info_classif
	# #determine the mutual information
	# mutual_info = mutual_info_classif(X_train, y_train)
	# mutual_info
	# mutual_info = pd.Series(mutual_info)
	# #mutual_info.index = X_train.columns
	# mutual_info.sort_values(ascending=False)
	# mutual_info.sort_values(ascending=False).plot.bar(figsize=(20,8))
