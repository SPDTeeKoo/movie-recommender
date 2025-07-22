
---

# Movie Recommendation System

A content-based movie recommender web app built with Streamlit. 
**[👉 Try the Deployed App Here](https://movie-recommender-4ysamwcbp6f2enaveqrrsp.streamlit.app/)**
This app suggests movies similar to your favorite one using advanced data analysis and natural language processing techniques.

## Features

- **Find movies similar to your favorite**: Select a movie and get top 10 recommendations.
- **Content-based filtering**: Uses movie overviews and metadata for recommendations.
- **Interactive web interface**: Built with Streamlit for ease of use.

---

## Data Sources

- [`tmdb_5000_movies.csv`](tmdb_5000_movies.csv): Movie metadata from TMDB.
- [`tmdb_5000_credits.csv`](tmdb_5000_credits.csv): Cast and crew information.

---

## Data Analysis & Preprocessing

All data analysis was performed in [`movie_recom.ipynb`](movie_recom.ipynb):

1. **Data Loading & Merging**
   - Loaded both movies and credits datasets.
   - Merged on the `id` column for a unified dataset.

2. **Data Cleaning**
   - Dropped irrelevant columns: `homepage`, `title_x`, `title_y`, `production_countries`, `status`.
   - Renamed columns for consistency.

3. **Outlier Removal**
   - Analyzed `vote_count` distribution.
   - Filtered movies with vote counts above the 90th percentile to focus on popular titles.

4. **Weighted Rating Calculation**
   - Used IMDB’s weighted rating formula:
     \[
     \text{Weighted Average} = \frac{(R \times V) + (C \times m)}{V + m}
     \]
     Where:
     - \( R \) = average rating for the movie
     - \( V \) = number of votes for the movie
     - \( C \) = mean vote across all movies
     - \( m \) = minimum votes required (90th percentile)

5. **Top Movies by Rating & Popularity**
   - Ranked movies by weighted average and popularity.
   - Visualized top movies using bar plots.

6. **Hybrid Scoring**
   - Normalized `weighted_average` and `popularity` using MinMaxScaler.
   - Created a hybrid score: `score_mix = 0.5 * weighted_average_scaled + 0.5 * popularity_scaled`.
   - Ranked movies by this hybrid score.

7. **Content-Based Recommendation**
   - Used the `overview` text for each movie.
   - Applied TF-IDF vectorization (with n-grams and English stopwords removal).
   - Computed pairwise similarity using the sigmoid kernel.
   - For a selected movie, recommended the top 10 most similar movies.

8. **Model & Data Export**
   - Saved processed data and models (`tfidf_vectorizer`, `sigmoid_kernel`) for use in the app.

---

## App Structure

- **`app.py`**: Streamlit app that loads preprocessed data and models, and provides the user interface.
- **`dumped_obj/`**: Contains preprocessed CSVs and serialized model objects.

### How it works

1. User selects a movie from the dropdown.
2. The app computes similarity scores using the sigmoid kernel on TF-IDF vectors.
3. Top 10 most similar movies are displayed.

---

## How to Run

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Start the app**:
   ```
   streamlit run app.py
   ```

3. **Open in browser**: Follow the Streamlit link to interact with the app.

---

## Methodology

- **Content-based filtering**: Recommends movies based on textual similarity of overviews.
- **Hybrid scoring**: Combines popularity and rating for better recommendations.
- **Data visualization**: Used for exploratory data analysis and feature engineering.

---

## Acknowledgements

- Data from [The Movie Database (TMDB)](https://www.themoviedb.org/).
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), and [pandas](https://pandas.pydata.org/).

---

Feel free to further customize this README for your needs! If you want the markdown as a file, let me know.

