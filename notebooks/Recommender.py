import streamlit as st
import pandas as pd
import numpy as np 
import warnings

# 📊 Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# 🤖 Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None) # display all columns
warnings.filterwarnings('ignore') # ignore warnings

# Recommender

books_final = pd.read_csv("books_final.csv")

data_features = books_final[["Author", "Genre"]]

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False) # To avoid having an sparse_matrix as output

ohe.fit(data_features[['Author','Genre']]) # The .fit() method determines the unique values of each column
data_features_ohe = ohe.transform(data_features[['Author','Genre']])
data_features_ohe = pd.DataFrame(data_features_ohe)

num_clusters = 7  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
books_final["cluster"] = kmeans.fit_predict(data_features_ohe)

def recommend_similar_books(book_index, df):
    cluster_label = df.loc[book_index, "cluster"]
    return df[df["cluster"] == cluster_label].sample(5)  # Return 5 random recommendations

# Title

st.title("Book Recommender")

# Sidebar Navigation
st.sidebar.title("Please choose what you would like to do")
menu = st.sidebar.radio("Select an option:", ["Recommendation based on Title", "Random Pick"])

if menu == "Recommendation based on Title":
    st.subheader("Enter your favourite book and get recommendations for your next read!")
    st.subheader("Please enter your favourite Book")

    options = books_final["Title"]

    choice = st.selectbox("Please enter your favourite Book:", options)

    book_index = books_final[books_final['Title'] == choice].index
    cluster_label = books_final.loc[book_index, "cluster"]
    cluster_value = cluster_label.iloc[0]
    rec = books_final[books_final["cluster"] == cluster_value].sample(5)
    rec = rec.reset_index(drop=True).drop(columns=rec.columns[-1])


    st.write("You selected:", choice)
    st.write(f"Your recommendations are:")
    st.dataframe(rec)
    

elif menu == "Random Pick":

    random = books_final.sample(5)
    random = random.reset_index(drop=True).drop(columns=random.columns[-1])
    
    st.subheader("Please press the button")
    st.write(f"Your recommendations are:")
    if st.button("Click me!"):
        st.dataframe(random)
    


