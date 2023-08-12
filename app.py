import streamlit as st
import pandas as pd
import pickle
with open('C:/Users/dell/Downloads/app.py', 'rb') as file:
    python_code = file.read()

pickled_code = pickle.dumps(python_code)

with open(C:/Users/dell/Downloads/app.pkl, 'wb') as file:
    file.write(pickled_code)

def main():
    st.title('Car Price Prediction Using ML')
    st.subheader('Car Price Predictor')
    st.info('''We need some information to predict Car price''')


    car_details = pd.read_csv('C:/Users/dell/Desktop/capstone/CAR DETAILS.csv')
    Brand=(car_details['name'].unique())
    transmission=(car_details['transmission'].unique())
    seller=(car_details['seller_type'].unique())
    owner=(car_details['owner'].unique())
    fuel=(car_details['fuel'].unique())

    p2=st.slider('Model Year',2005,2020,2005)

    p3=st.selectbox('Seller Type',seller)
    if p3=='Individual':
        p3=1
    elif p3=='Dealer':
        p3=0
    elif p3=='Trustmark Dealer':
        p3=2

    p4=st.selectbox('Owner Type',owner)
    if p4=='First Owner':
        p4=0
    elif p4=='Second Owner':
        p4=2
    elif p4=='Third Owner':
        p4=4
    elif p4=='Fourth & Above Owner':
        p4=1
    elif p4=='Test Drive Car':
        p4=3

    p5=st.selectbox('Transmission Type',transmission)
    if p5=='Manual':
        p5=1
    elif p5=='Automatic':
        p5=0

    p6=st.selectbox('Fuel Type',fuel)
    if p6=='Petrol':
        p6=4
    elif p6=='Diesel':
        p6=1
    elif p6=='CNG':
        p6=0
    elif p6=='LPG':
        p6=3
    elif p6=='Electric':
        p6=2

    p7=(st.slider('KM Driven',500,10000000,500))/100000

    x=pd.DataFrame({'year':[p2],'fuel':[p6],'seller_type':[p3],
                    'transmission':[p5],'owner':[p4],'km_driven_in_lacks':[p7]})
    ok=st.button('Predict Car Price')
    if ok:
        prediction=model.predict(x)
        st.success('Predicted Car Price:'+str( prediction*100000) +'Rupees')
        st.caption('Thanks for using!')
        st.balloons()
        st.write('Created by Lalit Yadav.')

if __name__=='__main__':
    main()
