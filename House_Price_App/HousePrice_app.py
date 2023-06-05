import streamlit as st
import pandas as pd

import xgboost as xgb
import plotly.express as px


st.set_page_config(page_title="House Prediction Tool")

header=st.container()
dataset=st.container()
filt=st.container()
result=st.container()



with header:
    st.title("House Price Prediction Tool")

with dataset:
    #st.header("Data Snapshot")
    x=pd.read_csv('realtor-data.csv')
    #st.write(x.head(5))
    
s=st.sidebar.selectbox("Select State",options=x['state'].unique())
bed=st.sidebar.slider("Specify # of bedrooms",min_value=1,max_value=6)
bath=st.sidebar.slider("# of baths",min_value=1,max_value=10)
house_size=st.sidebar.number_input("House Area (sq.ft)",value=1800.00)
lot_size=st.sidebar.number_input("Lot Size (acre)",value=0.05)
with filt:
    
    x1=x[x['state']==s]
    
    
    D=pd.DataFrame({"bed":[bed],"bath":[bath],"acre_lot":[lot_size],"house_size":[house_size]})
    #st.write(D)
    
    x1=x[x['state']==s]
    h=x1['price'].quantile(0.75)+1.5*(x1['price'].quantile(0.75)-x1['price'].quantile(0.25))
    x1=x1[x1['price']<h]
    x2=x1[['bed','bath','acre_lot','house_size','price']]
    x3=x2.dropna()
    x4=x3.reset_index(drop=True)
    
    model=xgb.XGBRegressor(max_depth=20)
    model.fit(x4.drop(['price'],axis=1),x4['price'])
    
    predicted_price=model.predict(D)
    p=pd.DataFrame(predicted_price)
    p.columns=['Predicted Price']
    
    
with result:
    #col1,col2=st.columns(2)
    p['Predicted Price'] = p['Predicted Price'].apply(lambda x: "${:.1f}k".format((x/1000)))
    st.text('Predicted Price:')
    
    #st.subheader('Predicted Price:')
    
    st.subheader(p.iloc[0,0])
    
    x5=x[(x['state']==s)&(x['bed']==bed)&(x['bath']==bath)]
    
    st.write("Price distribution for similar houses")
    fig=px.histogram(x5,x='price')
    st.plotly_chart(fig)
