import streamlit as st
import pyarrow.parquet as pq
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from validators import Min



siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()
modelTraining = st.container()

@st.cache
def get_data():
    df = pd.read_csv('tripdata.csv')
    return df


with siteHeader:
	st.title('Welcome to the Awesome project!')
	st.text('In this project I look into ... And I try ... I worked with the dataset from ...')

with dataExploration:
    st.header('Dataset: NY Cab dataset')
    st.text('I found this dataset at... I decided to work with it because ...')
    df = get_data()
    st.write(df.head())
    distribution_pickup = pd.DataFrame(df['PULocationID'].value_counts()).head(50) 
    st.bar_chart(distribution_pickup)



with newFeatures:
    st.header('New features I came up with')
    st.text('Let\'s take a look into the features I generated.')
    st.markdown('* **first feature:** this is the explanation')
    st.markdown('* **second feature:** another explanation')


with modelTraining:
    st.header('Model training')
    st.text('In this section you can select the hyperparameters!')


    max_depth = st.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    number_of_trees = st.selectbox('How many trees should there be?', 
      options=[100,200,300,'No limit'], 
      index=0)

    st.text('Here is a list of features: ')
    st.write(df.columns)
    input_feature = st.text_input('Which feature would you like to input to the model?', 
      'PULocationID')

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=number_of_trees)    #1

    X = df[[input_feature]]     #2
    y = df[['trip_distance']]
    y=y.values.reshape(-1, 1)     #3

    regr.fit(X, y) #4
    prediction = regr.predict(y) #5

    st.subheader('Mean absolute error:') #6
    st.write(mean_absolute_error(y, prediction)) #7

st.markdown(
      """
    <style>
     .main {
     background-color: #FA6B6D;

     }

</style>
      
      """,
      unsafe_allow_html=True
  )









































