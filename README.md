# Book-Recommender-ML

A Book Recommender System built with Streamlit that suggests books based on similarity. It combines web-scraped book data and Google Books API metadata with an unsupervised machine learning model (K-Means clustering) to provide personalized recommendations. The features that went into the model are author and genre due to an high amount of missing values for additional features.

Features

- Interactive Streamlit web app for book discovery
- Web scraping from Books to Scrape and Google Books API for dataset
- K-Means clustering for unsupervised grouping of similar books (based on author and genre)

Tech Stack

- Python (data processing & ML)
- Streamlit (UI/UX)
- BeautifulSoup (web scraping)
- Requests (API calls)
- Scikit-learn (K-Means clustering)
- Pandas / NumPy (data wrangling)

Machine Learning Approach

- Preprocessed book data (author, genre)
- Transforming categorical features with OneHotEncoder
- Calculated optimal amount of clusters with Elbow method
- Applied K-Means clustering to group similar books
- Recommender suggests books from the same cluster as the selected book title






