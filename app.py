import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.title("Recommendation System")
st.text("Let us help you")
st.image("0.jpg")

food = pd.read_csv("madchef.csv")
food = food.iloc[: , 1:]
ratings = pd.read_csv("madchef1.csv")
ratings = ratings.iloc[: , 1:]


st.subheader("What food do you prefer?")
cuisine = st.selectbox("Choose your buget!",food['Menu_Price'].unique())

# ratings = pd.read_csv("ratings.csv")
combined = pd.merge(ratings, food, on='index')


ans = combined.loc[(combined.Menu_Price == cuisine),['Menu_Price','Menu_Name']]
names = ans['Menu_Name'].tolist()

x = np.array(names)
ans1 = np.unique(x)

finallist = ""
bruh = st.checkbox("Choose your Dish")
if bruh == True:
    finallist = st.selectbox("Our choices Base on price",ans1)



##### IMPLEMENTING RECOMMENDER ######
dataset = ratings.pivot_table(index='index',columns='Menu_price_value',values='Menu_description_value')
dataset.fillna(0,inplace=True)
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

def food_recommendation(Food_Name):
    n = 3
    FoodList = food[food['Menu_Name'].str.contains(Food_Name)]  
    if len(FoodList):        
        Foodi= FoodList.iloc[0]['index']
        Foodi = dataset[dataset['index'] == Foodi].index[0]
        distances , indices = model.kneighbors(csr_dataset[Foodi],n_neighbors=n+1)    
        Food_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        Recommendations = []
        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['index']
            i = food[food['index'] == Foodi].index
            Recommendations.append({'Menu_Name':food.iloc[i]['Menu_Name'].values[0],'Distance':val[1]})
        df = pd.DataFrame(Recommendations,index=range(1,n+1))
        return df['Menu_Name']
    else:
        return "No Similar Foods."


display = food_recommendation(finallist)
#names1 = display['Name'].tolist()

#x1 = np.array(names)
#ans2 = np.unique(x1)
if bruh == True:
    bruh1 = st.checkbox("We also Recommend : ")
    if bruh1 == True:
        for i in display:
            st.write(i)
