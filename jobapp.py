import streamlit as st

import seaborn as sns
iris = sns.load_dataset('iris')
st.write(iris.tail())
import numpy as np

with st.form("Flower"):
    petal_length = st.number_input("Enter Petal Length",min_value=1.0,max_value=6.9, step=0.1 )
    petal_width = st.number_input("Enter Petal width", min_value=0.1, max_value=2.5, step=0.1 )
    sepal_length = st.number_input("Enter sepal Length", min_value=4.3, max_value=7.9, step=0.1)
    sepal_wdth = st.number_input("Enter sepal width", min_value= 2.0, max_value=4.4, step =0.1) 
    main = st.form_submit_button('click herr to enter values') 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
X = iris.drop('species', axis = 1)
y= iris.species
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knn.fit(X_train, y_train)

values = np.array([sepal_length,sepal_wdth, petal_length,petal_width]).reshape(1,-1)

if main:
    st.write(f" the flower is {knn.predict(values)[0]}")