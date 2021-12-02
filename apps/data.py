import streamlit as st
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from apps import home
import os
from dateutil import parser
def app():
  if 'main_data.csv' not in os.listdir('/content/skripsi/data'):
    st.markdown("Please upload data through `Home` page!")
  else:
    df = pd.read_csv('/content/skripsi/data/main_data.csv')
    var_enrolled = np.array(df['enrolled'])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.array(df.drop(labels=['enrolled'], axis=1)), var_enrolled, test_size=0.2, random_state=111)
    st.write(X_test)
    from sklearn.feature_selection import mutual_info_classif
    #determine the mutual information
    mutual_info = mutual_info_classif(X_train, y_train)
    mutual_info
    mutual_info = pd.Series(mutual_info)
    #mutual_info.index = X_train.columns
    mutual_info.sort_values(ascending=False)
    mutual_info.sort_values(ascending=False).plot.bar(title='urutannya')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    from sklearn.feature_selection import SelectKBest
    sel_five_cols = SelectKBest(mutual_info_classif, k=10)
    sel_five_cols.fit(X_train, y_train)
    #X_train.columns[sel_five_cols.get_support()]
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import f1_score
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB() # Define classifier)
    nb.fit(X_train, y_train)

    # Make predictions
    y_train_pred = nb.predict(X_train)
    y_test_pred = nb.predict(X_test)

    # Training set performance
    nb_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    nb_train_mcc = matthews_corrcoef(y_train, y_train_pred) # Calculate MCC
    nb_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score

    # Test set performance
    nb_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    nb_test_mcc = matthews_corrcoef(y_test, y_test_pred) # Calculate MCC
    nb_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score

    st.write('Model performance for Training set')
    st.write('- Accuracy: %s' % nb_train_accuracy)
    st.write('- MCC: %s' % nb_train_mcc)
    st.write('- F1 score: %s' % nb_train_f1)
    st.write('----------------------------------')
    st.write('Model performance for Test set')
    st.write('- Accuracy: %s' % nb_test_accuracy)
    st.write('- MCC: %s' % nb_test_mcc)
    st.write('- F1 score: %s' % nb_test_f1)

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=10) # Define classifier
    rf.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # Training set performance
    rf_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    rf_train_mcc = matthews_corrcoef(y_train, y_train_pred) # Calculate MCC
    rf_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score

    # Test set performance
    rf_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    rf_test_mcc = matthews_corrcoef(y_test, y_test_pred) # Calculate MCC
    rf_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score

    st.write('Model performance for Training set')
    st.write('- Accuracy: %s' % rf_train_accuracy)
    st.write('- MCC: %s' % rf_train_mcc)
    st.write('- F1 score: %s' % rf_train_f1)
    st.write('----------------------------------')
    st.write('Model performance for Test set')
    st.write('- Accuracy: %s' % rf_test_accuracy)
    st.write('- MCC: %s' % rf_test_mcc)
    st.write('- F1 score: %s' % rf_test_f1)

    from sklearn.ensemble import StackingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    estimator_list = [
        ('nb',nb),
        ('rf',rf)]

    # Build stack model
    stack_model = StackingClassifier(
        estimators=estimator_list, final_estimator=KNeighborsClassifier(3)
    )

    # Train stacked model
    stack_model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = stack_model.predict(X_train)
    y_test_pred = stack_model.predict(X_test)

    # Training set model performance
    stack_model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    stack_model_train_mcc = matthews_corrcoef(y_train, y_train_pred) # Calculate MCC
    stack_model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score

    # Test set model performance
    stack_model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    stack_model_test_mcc = matthews_corrcoef(y_test, y_test_pred) # Calculate MCC
    stack_model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score

    st.write('Model performance for Training set')
    st.write('- Accuracy: %s' % stack_model_train_accuracy)
    st.write('- MCC: %s' % stack_model_train_mcc)
    st.write('- F1 score: %s' % stack_model_train_f1)
    st.write('----------------------------------')
    st.write('Model performance for Test set')
    st.write('- Accuracy: %s' % stack_model_test_accuracy)
    st.write('- MCC: %s' % stack_model_test_mcc)
    st.write('- F1 score: %s' % stack_model_test_f1)

    acc_train_list = {'nb':nb_train_accuracy,
    'rf': rf_train_accuracy,
    'stack': stack_model_train_accuracy}

    mcc_train_list = {'nb':nb_train_mcc,
    'rf': rf_train_mcc,
    'stack': stack_model_train_mcc}

    f1_train_list = {'nb':nb_train_f1,
    'rf': rf_train_f1,
    'stack': stack_model_train_f1}

    import pandas as pd

    acc_df = pd.DataFrame.from_dict(acc_train_list, orient='index', columns=['Accuracy'])
    mcc_df = pd.DataFrame.from_dict(mcc_train_list, orient='index', columns=['MCC'])
    f1_df = pd.DataFrame.from_dict(f1_train_list, orient='index', columns=['F1'])
    df = pd.concat([acc_df, mcc_df, f1_df], axis=1)
    st.write(df)
