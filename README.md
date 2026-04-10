# 🎬 Movie Recommendation System

## Kaise chalayein (How to Run)

### Step 1 — Install requirements
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

Browser mein automatically open ho jayega: `http://localhost:8501`

---

## Project Structure
```
movie_rec/
├── app.py              ← Main Streamlit app
├── movies_df.pkl       ← Preprocessed movie data
├── tfidf_matrix.pkl    ← TF-IDF vectors (content-based)
├── title_idx.pkl       ← Title → index mapping
├── requirements.txt
└── README.md
```

---

## 🔬 Techniques Used

### 1. Content-Based Filtering
- Movie ke `overview` + `genres` ko TF-IDF vectors mein convert kiya
- Cosine Similarity se similar movies dhundhi jaati hain
- `max_features=10000` for TF-IDF

### 2. Popularity-Based (Weighted Rating)
- IMDB's Bayesian formula use ki:
  ```
  WR = (v/(v+m)) * R + (m/(v+m)) * C
  ```
  - `v` = number of votes for a movie
  - `m` = minimum votes required (70th percentile)
  - `R` = average rating of the movie
  - `C` = mean vote across all movies

### 3. Filters
- Genre, Year Range, Minimum Rating sab filter kar sakte ho

---

## 📊 Dataset
- Source: MovieLens / TMDB metadata (~45,000 movies)
- Fields used: title, overview, genres, vote_average, vote_count, release_date
