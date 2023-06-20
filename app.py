import streamlit as st 
import pandas as pd
import numpy as np
from prediction import predict


st.title("classifying iris flowers")
st.markdown('Toy model to play classify iris flowers into \
            (setosa,versicolor,virginica)based on their sepal/petal \
            and length/width.')

st.header("plant features")
col1,col2=st.columns(2)

with col1:
    st.text("sepal characteristics")
    sepal_l=st.slider('sepal length(cm)',1.0,8.0,0.5)
    sepal_w=st.slider('sepal width(cm)',2.0,4.4,0.5)

with col2:
    st.text("petal characteristics")
    petal_l=st.slider('petal length(cm)',1.0,7.0,0.5)
    petal_w=st.slider('petal width(cm)',0.1,2.5,0.5)

    st.text('')
    if st.button("predict type of Iris"):
        result=predict( np.array([[sepal_l,sepal_w,petal_l,petal_w]]))
        st.text(result[0])    
        

 