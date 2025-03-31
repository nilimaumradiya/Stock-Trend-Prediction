
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf 
import streamlit as st 
from keras.models import load_model 

start = '2013-01-01'
end = '2023-12-31'
st.title(':red[Stock Trend Prediction]')

user_input = st.text_input(':rainbow[**Enter any ticker**]','GOOG')
df = yf.download(user_input, start, end)
df.head()


#Describing the data

st.subheader(':gray[Data from 2013-2023]')
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: linear-gradient(to bottom left, #434343  0%, black 100%);
background-size: cover;
}
</style>
'''


st.markdown(page_bg_img, unsafe_allow_html=True)
st.write(df.describe())

#Visualizations
st.subheader(':gray[Closing Price]')
fig1 = plt.figure(figsize = (12,6))
plt.plot(df.Close, label = 'closing price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

st.subheader(':gray[Closing Price vs Time Chart 100MA]')
ma100 = df.Close.rolling(100).mean()
fig2 = plt.figure(figsize = (12,6))

plt.plot(ma100, label = 'MA100')
plt.plot(df.Close, label = 'closing price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader(':gray[Closing Price vs Time Chart 200MA]')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize = (12,6))
plt.plot(ma100,'r', label = 'MA100')
plt.plot(ma200,'g', label = 'MA200')
plt.plot(df.Close,'b', label = 'closing price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

#Splitting Data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

data_training_array = scaler.fit_transform(data_training)


#Load my model 
model = load_model('keras_model.h5')


#Testing Part
past_100_days = data_training.tail(100) 
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
 
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test = np.array(x_test)
y_test = np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph
st.subheader(':gray[Prediction] ')
fig4 = plt.figure(figsize=(12,6))
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

#Final Graph
st.subheader(':gray[Prediction vs Original] ')
fig4 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
