# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:01:03 2020

@author: Chamsedine
"""
#pd.version.version
import numpy as np
import streamlit as st
import os

# EDA Pkgs
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pyforest

# Data Viz Pkgs
import matplotlib
matplotlib.use('Agg')# To Prevent Errors
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import Counter
import matplotlib.ticker as mtick

# ML Pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

def main():
    """ ML App with Streamlit for Contraceptive Choice Prediction"""
    st.title("Application Of Machine Learning ")
    st.subheader("Data Understanding")
    
    st.markdown("""
        #### Description
        This is a simple Exploratory Data Analysis and build Machine Learning algorithme of the Churn Dataset.
        Our aim is to find and predict the customer who will move the Banque.
        We use Streamlit framework. 
        """)

    # Load Our Dataset
    df = pd.read_csv("Churn_Modelling.csv")
    new_df = df.copy
    
    

    if st.checkbox("Show DataSet"):
        st.write(df.head())
        #number = st.number_input("Number of Rows to View")
        #st.dataframe(df.head(number))
    if st.checkbox("Show all DataFrame"):
        st.dataframe(df)
        
    if st.button("Columns Names"):
        st.write(df.columns)
        
    if st.checkbox("Select Columns To Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect('Select',all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)
        
     # Show Dimensions and Shape of Dataset
    data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
    if data_dim == 'Rows':
        st.text("Showing Length of Rows")
        st.write(len(df))
    if data_dim == 'Columns':
        st.text("Showing Length of Columns")
        st.write(df.shape[1])
        
        
    st.subheader("Data Visualization")
    
    # Show Correlation Plots
   # if st.checkbox("Pie Plot"):
       #all_columns_names = df.columns.tolist()
        #st.info("Please Choose Target Column")
        #int_column =  st.selectbox('Select Int Columns For Pie Plot',all_columns_names)
    if st.button("Churn Plot"):
            # cust_values = df[int_column].value_counts()
            # st.write(cust_values.plot.pie(autopct="%1.1f%%"))
            st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()
            
    if st.button("Country By Client"):
        city_order = df['Geography'].unique()
        sns.countplot(x ='Geography', data = df, order = city_order)
        plt.xlabel('country')
        plt.ylabel("Number of churn by country")
        st.text("The majority of this bank’s clients come from France")
        st.pyplot()
        
    if st.button("Churn By Stat"):
        
        pd.crosstab(df["Geography"],df["Exited"]).plot(kind='bar')
        plt.title('Churn by country')
        plt.xlabel('Country')
        plt.ylabel('Fréquence')
        st.pyplot()
   
        
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder() 
    df['Surname'] = lb.fit_transform(df['Surname'])
    df['Gender'] = lb.fit_transform(df['Gender'])
    df['Geography'] = lb.fit_transform(df['Geography'])
        
     # Show Summary of Dataset
    #if st.checkbox("Show Summary of Dataset"):
        #st.write(df.describe())
        
    if st.checkbox("Show after Data processing"):
        st.write(df.head())
        
     #if st.checkbox("Valeurs Uniques"):
        #for var in df.columns:
            #st.dataframe(df[var].unique())
            
    if st.button("Data Types"):
        st.write(df.dtypes)
    
    
        
    #if st.button("Country Values"):
       #st.write(df['Geography'].value_counts())
       
    #if st.button("Counts Churn"):
        #st.text("Value Counts By Target/Class")
        #st.write(df.iloc[:,-1].value_counts())

      
    #if st.button("Numeric Variable"):
       #st.write([var for var in df.columns if df[var].dtypes != 'O'])
       
    #if st.button("Variables categoriques"):
       #st.write([var for var in df.columns if df[var].dtypes == 'O'])
       
    #if st.checkbox("Stats Churn"):
       #st.text("Percentage the customer to leave")
       #st.write(round(len(df[df['Exited'] == 1]) / 10000 * 100,2))
       #st.text("Percentage the customer to stay")
       #st.write(round(len(df[df['Exited'] == 0]) / 10000 * 100,2))
      
    
    if st.checkbox("Check the Nan Values"):
        st.text("Incredible!!!!! our dataset is clean.However, in the real world, a clean dataset is rare to find")
        st.write(pd.isnull(df).sum())
        

    #if st.checkbox("Select Columns To Show"):
        #all_columns = df.columns.tolist()
        #selected_columns = st.multiselect('Select',all_columns)
        #new_df = df[selected_columns]
        #st.dataframe(new_df)

   
    # Prediction
    st.subheader("Build Machine Learning Algorythme")
    #Importing the Library
    
    from sklearn.tree import DecisionTreeClassifier
    
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    
    
    X = df.iloc[:,0:13].values
    y = df.iloc[:,13].values

 
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    
    #Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test =  sc_X.transform(X_test)
    
    
    alg = ['Decision Tree', 'Support Vector Machine', 'Logistic Regression']
    classifier = st.selectbox('Which algorithm?', alg)
    if classifier=='Decision Tree':
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        acc = dtc.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = dtc.predict(X_test)
        #st.write('Accuracy test: ', acc)
        cm_dtc=confusion_matrix(y_test,pred_dtc)
        st.write('Confusion matrix: ', cm_dtc)
        
    elif classifier == 'Support Vector Machine':
        svm=SVC()
        svm.fit(X_train, y_train)
        acc_svm = svm.score(X_test, y_test)
        st.write('Accuracy: ', acc_svm)
        pred_svm = svm.predict(X_test)
        cm_svm=confusion_matrix(y_test,pred_svm)
        st.write('Confusion matrix: ', cm_svm)
        
    elif classifier == 'Logistic Regression':
        lr=LogisticRegression(solver='lbfgs', random_state = 0)
        lr.fit(X_train, y_train)
        acc_log = lr.score(X_test, y_test)
        st.write('Accuracy: ', acc_log)
        pred_lr = lr.predict(X_test)
        cm_log=confusion_matrix(y_test,pred_lr)
        st.write('Confusion matrix: ', cm_log)
        
    #elif classifier == 'KNeighbors Classifier':
        #knn=KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        #knn.fit(X_train, y_train)
        #acc_knn = knn.score(X_test, y_test)
        #st.write('Accuracy: ', acc_knn)
        #pred_knn = knn.predict(X_test)
        #cm_knn=confusion_matrix(y_test,pred_knn)
        #st.write('Confusion matrix: ', cm_knn)
        
        
    st.markdown("""
        Conclusion:
        Based on its results, we can conclude that the best model is the tree of
            decision despite its lower score than the other algos.
        However, this is just a demonstration, these models can be widely 
        improved

        """)
    

#if __name__ == '__main__':
	#main()
    
    st.sidebar.subheader("About")
    #st.sidebar.info("ML App with Streamlit")
    #st.sidebar.text("Streamlit Is Awesome")
    if st.sidebar.button("About"):
        st.sidebar.text("AIDARA Chamsedine")
        st.sidebar.text("aidarachamsedine10@gmail.com")

if __name__ == '__main__':
    main()
