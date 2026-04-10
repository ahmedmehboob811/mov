import streamlit as st
import pandas as pd
import numpy as np
import ast, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎬 Movie Recommender AI", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .main-header{font-size:2.6rem;font-weight:800;background:linear-gradient(135deg,#e50914,#ff6b6b);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:.2rem}
  .sub-header{text-align:center;color:#888;font-size:.95rem;margin-bottom:1.5rem}
  .movie-card{background:#1a1a2e;border-radius:12px;padding:1rem;margin:.4rem 0;border-left:4px solid #e50914}
  .movie-title{font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:.2rem}
  .movie-meta{font-size:.82rem;color:#aaa}
  .badge{display:inline-block;background:#e50914;color:#fff;padding:2px 8px;border-radius:20px;font-size:.72rem;margin-right:4px}
  .tag{display:inline-block;background:#0f3460;color:#e94560;padding:2px 8px;border-radius:20px;font-size:.72rem;margin-right:4px;border:1px solid #e94560}
  .star{color:#f5c518}
  .section-title{font-size:1.3rem;font-weight:700;color:#e50914;margin:1rem 0 .5rem;border-bottom:2px solid #333;padding-bottom:.3rem}
  .cluster-card{background:#0f3460;border-radius:10px;padding:.8rem;margin:.3rem 0;border:1px solid #e94560}
  .pill-pos{background:#1b5e20;color:#a5d6a7;padding:3px 10px;border-radius:20px;font-size:.75rem}
  .pill-neg{background:#b71c1c;color:#ffcdd2;padding:3px 10px;border-radius:20px;font-size:.75rem}
  .pill-neu{background:#33333a;color:#bbb;padding:3px 10px;border-radius:20px;font-size:.75rem}
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "movies_metadata.csv")
    if not os.path.exists(csv_path):
        st.error("movies_metadata.csv not found!"); st.stop()

    df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)

    def to_float(series):
        out = np.empty(len(series), dtype=float)
        for i, v in enumerate(series):
            try:    out[i] = float(v)
            except: out[i] = np.nan
        return out

    df['vote_average'] = to_float(df['vote_average'])
    df['vote_count']   = to_float(df['vote_count'])
    df['popularity']   = to_float(df['popularity'])
    df = df.dropna(subset=['title','overview','vote_average','vote_count']).reset_index(drop=True)

    def extract_genres(x):
        try:    return ' '.join(g['name'] for g in ast.literal_eval(x))
        except: return ''
    df['genre_str'] = df['genres'].apply(extract_genres)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    v = df['vote_count'].values; R = df['vote_average'].values
    C = float(np.nanmean(R));    m = float(np.nanpercentile(v, 70))
    df['weighted_rating'] = (v/(v+m))*R + (m/(v+m))*C

    df['soup'] = df['overview'].fillna('') + ' ' + df['genre_str']
    return df

@st.cache_resource
def build_tfidf(soup_series):
    tfidf = TfidfVectorizer(stop_words='english', max_features=15000, ngram_range=(1,2))
    matrix = tfidf.fit_transform(soup_series)
    idx    = pd.Series(range(len(soup_series)), index=soup_series.index)
    return tfidf, matrix, idx

@st.cache_resource
def build_sentiment(_df):
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(str(t)) for t in _df['overview']]
    compound = np.array([s['compound'] for s in scores])
    pos      = np.array([s['pos']      for s in scores])
    neg      = np.array([s['neg']      for s in scores])
    return compound, pos, neg

@st.cache_resource
def build_clusters(_matrix, n_clusters=12):
    svd   = TruncatedSVD(n_components=50, random_state=42)
    reduced = svd.fit_transform(_matrix)
    km    = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(reduced)
    # 2D for scatter
    svd2  = TruncatedSVD(n_components=2, random_state=42)
    coords = svd2.fit_transform(_matrix)
    return labels, coords, reduced

@st.cache_resource
def build_keyword_index(_tfidf, _matrix):
    feature_names = np.array(_tfidf.get_feature_names_out())
    return feature_names

@st.cache_resource
def get_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_semantic_embeddings(_model, _overviews):
    return _model.encode(_overviews, batch_size=256, show_progress_bar=False)


with st.spinner("⏳ Loading movies..."):
    df = load_data()

with st.spinner("⚙️ Building TF-IDF index..."):
    tfidf_model, tfidf_matrix, title_idx_raw = build_tfidf(df['soup'])
    title_idx = pd.Series(df.index, index=df['title']).drop_duplicates()

ALL_GENRES = sorted(set(g.strip() for gs in df['genre_str'].dropna() for g in gs.split() if g))

CLUSTER_NAMES = {
    0:"Action & War", 1:"Romance & Drama", 2:"Horror & Thriller",
    3:"Sci-Fi & Space", 4:"Comedy & Family", 5:"Crime & Mystery",
    6:"Animation & Kids", 7:"Biography & History", 8:"Adventure & Fantasy",
    9:"Music & Arts", 10:"Documentary", 11:"International"
}

# ── Helpers ───────────────────────────────────────────────────
def content_recs(title, n=10):
    if title not in title_idx: return pd.DataFrame()
    idx = int(title_idx[title])
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = 0
    top = np.argsort(sims)[::-1][:n]
    res = df.iloc[top].copy(); res['similarity'] = sims[top]
    return res

def popular_movies(genre=None, yr_from=None, yr_to=None, min_r=0, n=20):
    f = df.copy()
    if genre:   f = f[f['genre_str'].str.contains(genre, na=False, case=False)]
    if yr_from: f = f[f['year'] >= yr_from]
    if yr_to:   f = f[f['year'] <= yr_to]
    f = f[f['vote_average'] >= min_r].dropna(subset=['weighted_rating'])
    return f.sort_values('weighted_rating', ascending=False).head(n)

def render_card(row, show_sim=False, show_sentiment=False):
    genres = row.get('genre_str','') or ''
    badges = ' '.join(f'<span class="badge">{g}</span>' for g in genres.split()[:3])
    try:    year   = int(row['year'])
    except: year   = 'N/A'
    try:    rating = round(float(row['vote_average']),1)
    except: rating = 'N/A'
    try:    votes  = int(row['vote_count'])
    except: votes  = 0
    sim_html = f'<span style="color:#4caf50;font-weight:700">{round(float(row["similarity"])*100,1)}% match</span> · ' if show_sim and 'similarity' in row else ''
    sent_html = ''
    if show_sentiment and 'compound' in row:
        c = float(row['compound'])
        if   c >= 0.05:  sent_html = f'<span class="pill-pos">😊 Positive</span> '
        elif c <= -0.05: sent_html = f'<span class="pill-neg">😰 Dark/Intense</span> '
        else:            sent_html = f'<span class="pill-neu">😐 Neutral</span> '
    overview = str(row.get('overview',''))[:180]+'...' if row.get('overview') else ''
    kw_html = ''
    if 'keywords' in row:
        kw_html = '<div style="margin-top:.4rem">' + ' '.join(f'<span class="tag">{k}</span>' for k in str(row['keywords']).split(',')[:5]) + '</div>'
    st.markdown(f"""
    <div class="movie-card">
      <div class="movie-title">🎬 {row['title']}</div>
      <div class="movie-meta">{sim_html}{sent_html}<span class="star">★</span> {rating} &nbsp;|&nbsp; {year} &nbsp;|&nbsp; {votes:,} votes</div>
      <div style="margin:.4rem 0">{badges}</div>
      {kw_html}
      <div style="font-size:.8rem;color:#ccc">{overview}</div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Navigation")
    page = st.radio("", [
        "🏠 Home",
        "🔍 Content-Based (TF-IDF)",
        "🏆 Top Rated",
        "🧠 NLP Analysis",
        "🤖 ML Clustering",
        "🔮 Semantic Search (DL)",
        "📊 Explore Data"
    ])
    st.markdown("---")
    vals = df['vote_average'].dropna().values
    st.markdown(f"**{len(df):,}** movies loaded")
    st.markdown(f"**{len(ALL_GENRES)}** genres · **{int(np.nanmin(df['year'].values))}-{int(np.nanmax(df['year'].values))}**")


# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🎬 Movie Recommender AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">TF-IDF · Sentiment NLP · KMeans Clustering · Sentence Transformers DL</div>', unsafe_allow_html=True)

    vals   = df['vote_average'].dropna().values
    yr_arr = df['year'].dropna().values
    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val in zip([c1,c2,c3,c4],
        ["🎥 Movies","⭐ Avg Rating","🎭 Genres","📅 Years"],
        [f"{len(df):,}", f"{float(np.mean(vals)):.1f}/10", str(len(ALL_GENRES)), f"{int(np.min(yr_arr))}-{int(np.max(yr_arr))}"]):
        col.metric(lbl, val)

    st.markdown("---")
    st.markdown("### 🔥 Quick Recommendations")
    quick = st.selectbox("Pick a movie:", [""]+sorted(df['title'].dropna().unique().tolist()), key="home_quick")
    if quick:
        recs = content_recs(quick, 6)
        if not recs.empty:
            cols = st.columns(2)
            for i,(_,row) in enumerate(recs.iterrows()):
                with cols[i%2]: render_card(row, show_sim=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🏆 All-Time Best")
        for _,row in popular_movies(n=5).iterrows(): render_card(row)
    with col_b:
        st.markdown("### 🔥 Most Popular")
        pop = df.dropna(subset=['popularity']).sort_values('popularity', ascending=False).head(5)
        for _,row in pop.iterrows(): render_card(row)


# ══════════════════════════════════════════════════════════════
# PAGE: CONTENT-BASED
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Content-Based (TF-IDF)":
    st.markdown('<div class="main-header">🔍 Content-Based Filtering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">TF-IDF bigrams (1-2 grams) on plot + genres → Cosine Similarity</div>', unsafe_allow_html=True)

    ca,cb = st.columns([3,1])
    with ca: movie_input = st.selectbox("Select movie:", [""]+sorted(df['title'].dropna().unique().tolist()))
    with cb: n_recs = st.slider("Results", 5, 20, 10)

    if movie_input:
        sel = df[df['title']==movie_input].iloc[0]
        st.markdown("#### 🎯 Selected Movie"); render_card(sel)
        recs = content_recs(movie_input, n_recs)
        st.markdown(f'<div class="section-title">Top {n_recs} Similar Movies</div>', unsafe_allow_html=True)
        if recs.empty: st.error("Not found in index.")
        else:
            cols = st.columns(2)
            for i,(_,row) in enumerate(recs.iterrows()):
                with cols[i%2]: render_card(row, show_sim=True)

    with st.expander("📖 How TF-IDF works"):
        st.markdown("""
        **TF-IDF (Term Frequency – Inverse Document Frequency)**
        - **TF**: kitni baar word ek document mein aaya
        - **IDF**: jo word har jagah common hai (like "the") uska weight kam karta hai
        - **Result**: har movie ka high-dimensional vector banta hai
        - **Cosine Similarity**: do vectors ka angle measure karta hai → closer = more similar
        - Is app mein **bigrams (1-2 word phrases)** use hote hain, sirf unigrams se better
        """)


# ══════════════════════════════════════════════════════════════
# PAGE: TOP RATED
# ══════════════════════════════════════════════════════════════
elif page == "🏆 Top Rated":
    st.markdown('<div class="main-header">🏆 Top Rated Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">IMDB Bayesian Weighted Rating: WR = (v÷(v+m))×R + (m÷(m+v))×C</div>', unsafe_allow_html=True)

    f1,f2,f3,f4 = st.columns(4)
    with f1: gs = st.selectbox("Genre", ["All"]+ALL_GENRES)
    with f2: yr = st.slider("Year Range", 1900, 2020, (1990,2020))
    with f3: mr = st.slider("Min Rating", 0.0, 10.0, 6.0, step=0.5)
    with f4: nt = st.slider("Top N", 10, 50, 20)

    res = popular_movies(None if gs=="All" else gs, yr[0], yr[1], mr, nt)
    st.markdown(f'<div class="section-title">Top {len(res)} Movies</div>', unsafe_allow_html=True)
    if res.empty: st.warning("No movies found!")
    else:
        cols = st.columns(2)
        for i,(_,row) in enumerate(res.iterrows()):
            with cols[i%2]: render_card(row)


# ══════════════════════════════════════════════════════════════
# PAGE: NLP ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🧠 NLP Analysis":
    st.markdown('<div class="main-header">🧠 NLP Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">VADER Sentiment · Keyword Extraction · Mood-Based Recommendations</div>', unsafe_allow_html=True)

    with st.spinner("Running VADER sentiment on 44k overviews..."):
        compound, pos_s, neg_s = build_sentiment(df)

    df_nlp = df.copy()
    df_nlp['compound'] = compound
    df_nlp['pos_score'] = pos_s
    df_nlp['neg_score'] = neg_s

    tab1, tab2, tab3 = st.tabs(["😊 Mood-Based Recs", "🔑 Keyword Extractor", "📊 Sentiment Stats"])

    # ── Tab 1: Mood-Based ────────────────────────────────────
    with tab1:
        st.markdown("### Filter movies by emotional tone of plot")
        mood = st.radio("What mood are you in?",
            ["😊 Feel-Good (Positive)", "😰 Dark & Intense (Negative)", "🎭 Balanced (Neutral)"],
            horizontal=True)
        
        col_g, col_y, col_n = st.columns(3)
        with col_g: genre_mood = st.selectbox("Genre filter", ["All"]+ALL_GENRES, key="mood_g")
        with col_y: yr_mood    = st.slider("Year", 1950, 2020, (2000,2020), key="mood_y")
        with col_n: n_mood     = st.slider("Results", 5, 30, 12, key="mood_n")

        if   "Positive" in mood:
            mask = df_nlp['compound'] >= 0.3
        elif "Negative" in mood:
            mask = df_nlp['compound'] <= -0.1
        else:
            mask = (df_nlp['compound'] > -0.1) & (df_nlp['compound'] < 0.3)

        mood_df = df_nlp[mask].copy()
        if genre_mood != "All":
            mood_df = mood_df[mood_df['genre_str'].str.contains(genre_mood, na=False, case=False)]
        mood_df = mood_df[(mood_df['year'] >= yr_mood[0]) & (mood_df['year'] <= yr_mood[1])]
        mood_df = mood_df.dropna(subset=['weighted_rating']).sort_values('weighted_rating', ascending=False).head(n_mood)

        st.markdown(f"**{len(mood_df)} movies found**")
        cols = st.columns(2)
        for i,(_,row) in enumerate(mood_df.iterrows()):
            with cols[i%2]: render_card(row, show_sentiment=True)

    # ── Tab 2: Keyword Extractor ─────────────────────────────
    with tab2:
        st.markdown("### Top TF-IDF Keywords per Movie")
        feature_names = build_keyword_index(tfidf_model, tfidf_matrix)

        kw_movie = st.selectbox("Pick a movie to extract keywords:",
            [""]+sorted(df['title'].dropna().unique().tolist()), key="kw_movie")
        if kw_movie and kw_movie in title_idx:
            idx = int(title_idx[kw_movie])
            row_vec = tfidf_matrix[idx].toarray().flatten()
            top_kw_idx = np.argsort(row_vec)[::-1][:15]
            top_kw = [(feature_names[i], round(float(row_vec[i]),4)) for i in top_kw_idx if row_vec[i] > 0]

            st.markdown(f"#### Keywords for **{kw_movie}**")
            kw_tags = ' '.join(f'<span class="tag">{k} ({s})</span>' for k,s in top_kw)
            st.markdown(f'<div style="line-height:2.2">{kw_tags}</div>', unsafe_allow_html=True)

            st.markdown("#### Find similar movies based on these keywords")
            recs = content_recs(kw_movie, 6)
            if not recs.empty:
                cols = st.columns(2)
                for i,(_,row) in enumerate(recs.iterrows()):
                    with cols[i%2]: render_card(row, show_sim=True)

    # ── Tab 3: Stats ─────────────────────────────────────────
    with tab3:
        st.markdown("### Sentiment Distribution of Movie Plots")
        pos_count = int((compound >= 0.05).sum())
        neg_count = int((compound <= -0.05).sum())
        neu_count = int(len(compound) - pos_count - neg_count)
        
        c1,c2,c3 = st.columns(3)
        c1.metric("😊 Positive Plots", f"{pos_count:,}", f"{pos_count/len(compound)*100:.1f}%")
        c2.metric("😐 Neutral Plots",  f"{neu_count:,}", f"{neu_count/len(compound)*100:.1f}%")
        c3.metric("😰 Negative Plots", f"{neg_count:,}", f"{neg_count/len(compound)*100:.1f}%")

        st.markdown("### Sentiment Score Distribution")
        hist_vals, hist_edges = np.histogram(compound, bins=40)
        hist_df = pd.DataFrame({'Sentiment Score': [round((hist_edges[i]+hist_edges[i+1])/2,2) for i in range(len(hist_vals))],
                                'Count': hist_vals}).set_index('Sentiment Score')
        st.bar_chart(hist_df)

        st.markdown("### Most Positive Movies (by plot)")
        top_pos = df_nlp.nlargest(5, 'compound')[['title','year','genre_str','compound']]
        st.dataframe(top_pos, use_container_width=True)

        st.markdown("### Most Dark/Intense Movies (by plot)")
        top_neg = df_nlp.nsmallest(5, 'compound')[['title','year','genre_str','compound']]
        st.dataframe(top_neg, use_container_width=True)

    with st.expander("📖 How VADER works"):
        st.markdown("""
        **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
        - Social media ke liye specifically designed NLP tool
        - Rule-based: punctuation, capitalization, intensifiers sab consider karta hai
        - **Compound score**: -1 (most negative) to +1 (most positive)
        - No training needed — pre-built lexicon use karta hai
        - Movie plot mein positive words (adventure, love, joy) → positive score
        - Dark themes (murder, death, violence) → negative score
        """)


# ══════════════════════════════════════════════════════════════
# PAGE: ML CLUSTERING
# ══════════════════════════════════════════════════════════════
elif page == "🤖 ML Clustering":
    st.markdown('<div class="main-header">🤖 ML Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">TruncatedSVD dimensionality reduction → KMeans clustering → Movie groups</div>', unsafe_allow_html=True)

    n_clusters = st.slider("Number of clusters (K)", 5, 20, 12)
    
    with st.spinner(f"Running KMeans with K={n_clusters}..."):
        labels, coords_2d, reduced = build_clusters(tfidf_matrix, n_clusters)

    df_cl = df.copy()
    df_cl['cluster'] = labels
    df_cl['x'] = coords_2d[:,0]
    df_cl['y'] = coords_2d[:,1]

    tab1, tab2, tab3 = st.tabs(["🗺️ Cluster Explorer", "📊 Cluster Visualization", "🔍 Find by Cluster"])

    # ── Tab 1: Explorer ──────────────────────────────────────
    with tab1:
        st.markdown("### Browse movies by cluster")
        cluster_id = st.selectbox("Pick a cluster:",
            [f"Cluster {i}" for i in range(n_clusters)], key="cl_pick")
        cid = int(cluster_id.split()[-1])

        cluster_movies = df_cl[df_cl['cluster']==cid].dropna(subset=['weighted_rating'])\
                            .sort_values('weighted_rating', ascending=False)

        # Auto-label cluster from top genres
        top_genres = ' '.join(cluster_movies['genre_str'].fillna('').head(20).values)
        genre_words = [g for g in top_genres.split() if g]
        from collections import Counter
        top_g = Counter(genre_words).most_common(3)
        auto_label = ' & '.join(g for g,_ in top_g) if top_g else f"Cluster {cid}"

        st.markdown(f"#### 🏷️ Cluster {cid}: **{auto_label}**")
        st.markdown(f"**{len(cluster_movies):,} movies** in this cluster")

        cols = st.columns(2)
        for i,(_,row) in enumerate(cluster_movies.head(10).iterrows()):
            with cols[i%2]: render_card(row)

    # ── Tab 2: Scatter Viz ───────────────────────────────────
    with tab2:
        st.markdown("### 2D PCA Scatter Plot of Movie Clusters")
        st.markdown("*SVD reduces 15,000-dim TF-IDF space to 2D for visualization*")

        sample_size = min(3000, len(df_cl))
        sample = df_cl.sample(sample_size, random_state=42)

        # Build scatter data per cluster
        scatter_data = {}
        for cid in range(n_clusters):
            mask = sample['cluster'] == cid
            if mask.sum() > 0:
                scatter_data[f"C{cid}"] = {
                    'x': sample[mask]['x'].tolist(),
                    'y': sample[mask]['y'].tolist(),
                }

        # Render as simple chart using streamlit
        chart_rows = []
        for cid in range(n_clusters):
            mask = sample['cluster'] == cid
            if mask.sum() > 0:
                for _, r in sample[mask].iterrows():
                    chart_rows.append({'x': float(r['x']), 'Cluster': f"C{cid}", 'y': float(r['y'])})
        chart_df = pd.DataFrame(chart_rows)

        # Use altair if available, else show table
        try:
            import altair as alt
            chart = alt.Chart(chart_df).mark_circle(size=20, opacity=0.5).encode(
                x=alt.X('x:Q', title='SVD Component 1'),
                y=alt.Y('y:Q', title='SVD Component 2'),
                color=alt.Color('Cluster:N', scale=alt.Scale(scheme='tableau20')),
                tooltip=['Cluster']
            ).properties(width=700, height=450, title='Movie Clusters (2D SVD Projection)')
            st.altair_chart(chart, use_container_width=True)
        except:
            st.dataframe(chart_df.groupby('Cluster').size().rename('Movie Count'), use_container_width=True)

        st.markdown("### Cluster Sizes")
        size_df = df_cl.groupby('cluster').size().reset_index(columns=['cluster','Count'] if False else None)
        size_df.columns = ['Cluster','Count']
        st.bar_chart(size_df.set_index('Cluster'))

    # ── Tab 3: Find cluster of a movie ──────────────────────
    with tab3:
        st.markdown("### Which cluster does a movie belong to?")
        cl_movie = st.selectbox("Search movie:", [""]+sorted(df['title'].dropna().unique().tolist()), key="cl_movie")
        if cl_movie and cl_movie in title_idx:
            idx  = int(title_idx[cl_movie])
            cid  = int(df_cl.iloc[idx]['cluster'])

            top_g_list = Counter(
                df_cl[df_cl['cluster']==cid]['genre_str'].fillna('').str.split().explode().tolist()
            ).most_common(3)
            label = ' & '.join(g for g,_ in top_g_list)

            st.success(f"**{cl_movie}** → Cluster **{cid}** ({label})")
            st.markdown("#### Other movies in the same cluster:")
            cluster_peers = df_cl[(df_cl['cluster']==cid) & (df_cl['title']!=cl_movie)]\
                .dropna(subset=['weighted_rating']).sort_values('weighted_rating',ascending=False).head(8)
            cols = st.columns(2)
            for i,(_,row) in enumerate(cluster_peers.iterrows()):
                with cols[i%2]: render_card(row)

    with st.expander("📖 How KMeans Clustering works"):
        st.markdown("""
        **Pipeline:**
        1. **TF-IDF Matrix** → 44k movies × 15k features (very sparse)
        2. **TruncatedSVD** → reduce to 50 dimensions (like PCA for sparse data)
        3. **KMeans** → K centroids find karta hai, har movie ko nearest centroid assign karta hai
        4. **Visualization** → SVD se 2D mein project karke scatter plot banaya

        **Inertia (within-cluster sum of squares)** kam hona better clustering ka sign hai.
        K choosy karna ek art hai — **Elbow Method** se optimal K dhundha ja sakta hai.
        """)


# ══════════════════════════════════════════════════════════════
# PAGE: SEMANTIC SEARCH (DL)
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Semantic Search (DL)":
    st.markdown('<div class="main-header">🔮 Semantic Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sentence Transformers (all-MiniLM-L6-v2) · 384-dim dense embeddings · Cosine Similarity</div>', unsafe_allow_html=True)

    st.info("⚠️ First load downloads ~80MB model from HuggingFace. Baad mein cache ho jaata hai.")

    if st.button("🚀 Load Deep Learning Model", type="primary"):
        with st.spinner("Downloading sentence-transformers model (~80MB)..."):
            model = get_sentence_model()
        with st.spinner("Encoding 44k movie plots... (2-3 minutes first time)"):
            overviews = df['overview'].fillna('').tolist()
            embeddings = build_semantic_embeddings(model, overviews)
        st.session_state['dl_ready'] = True
        st.session_state['dl_model'] = model
        st.session_state['dl_embeddings'] = embeddings
        st.success("✅ Model ready!")

    if st.session_state.get('dl_ready'):
        model      = st.session_state['dl_model']
        embeddings = st.session_state['dl_embeddings']

        tab1, tab2 = st.tabs(["💬 Describe & Find", "🎬 Deep Movie Similarity"])

        # ── Tab 1: Text Query ────────────────────────────────
        with tab1:
            st.markdown("### Describe what you want to watch — in your own words")
            examples = [
                "A hero fights aliens to save humanity",
                "Forbidden love between two people from different worlds",
                "A detective solving mysterious murders in a dark city",
                "Funny movie about family road trip gone wrong",
                "Psychological thriller where nothing is what it seems"
            ]
            ex = st.selectbox("Or pick an example:", [""]+examples)
            query = st.text_area("Your description:", value=ex, height=80,
                placeholder="e.g. 'a scientist travels through time to fix his mistakes'")
            n_dl = st.slider("Results", 5, 20, 10, key="dl_n")

            if query and st.button("🔍 Find Movies", type="primary"):
                with st.spinner("Encoding your query..."):
                    q_embed = model.encode([query])
                sims = cosine_similarity(q_embed, embeddings).flatten()
                top  = np.argsort(sims)[::-1][:n_dl]
                results = df.iloc[top].copy()
                results['similarity'] = sims[top]

                st.markdown(f'<div class="section-title">Top {n_dl} matches for: "{query}"</div>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i,(_,row) in enumerate(results.iterrows()):
                    with cols[i%2]: render_card(row, show_sim=True)

        # ── Tab 2: Deep Movie Similarity ─────────────────────
        with tab2:
            st.markdown("### Find semantically similar movies (DL vs TF-IDF comparison)")
            dl_movie = st.selectbox("Select movie:", [""]+sorted(df['title'].dropna().unique().tolist()), key="dl_movie")
            n_comp   = st.slider("Results", 5, 15, 8, key="dl_comp")

            if dl_movie and dl_movie in title_idx:
                idx = int(title_idx[dl_movie])

                # DL similarities
                movie_embed = embeddings[idx:idx+1]
                dl_sims = cosine_similarity(movie_embed, embeddings).flatten()
                dl_sims[idx] = 0
                dl_top = np.argsort(dl_sims)[::-1][:n_comp]
                dl_res = df.iloc[dl_top].copy(); dl_res['similarity'] = dl_sims[dl_top]

                # TF-IDF similarities
                tfidf_sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                tfidf_sims[idx] = 0
                tf_top = np.argsort(tfidf_sims)[::-1][:n_comp]
                tf_res = df.iloc[tf_top].copy(); tf_res['similarity'] = tfidf_sims[tf_top]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("#### 🔮 DL Semantic (Sentence Transformer)")
                    for _,row in dl_res.iterrows(): render_card(row, show_sim=True)
                with col_b:
                    st.markdown("#### 🔍 TF-IDF (Keyword Match)")
                    for _,row in tf_res.iterrows(): render_card(row, show_sim=True)
    else:
        st.markdown("""
        ### How Semantic Search works differently from TF-IDF:

        | | TF-IDF | Sentence Transformers |
        |--|--|--|
        | Type | Sparse keyword vectors | Dense neural embeddings |
        | "happy" ≈ "joyful"? | ❌ No | ✅ Yes |
        | Context aware? | ❌ No | ✅ Yes |
        | Speed | ⚡ Instant | 🐌 Slower |
        | Model size | None | ~80MB |

        Sentence Transformers (BERT-based) **understand meaning**, not just keywords.
        """)

    with st.expander("📖 How Sentence Transformers work"):
        st.markdown("""
        **all-MiniLM-L6-v2** — distilled BERT model, 6 layers, 384-dim embeddings

        1. Movie ka overview → Transformer model → 384-dimensional vector
        2. Query text bhi same model se encode hota hai
        3. **Cosine similarity** se closest movie plots dhundhe jaate hain
        4. TF-IDF se fark: "spacecraft" aur "rocket ship" same movies return karega
           kyunki model semantic meaning samajhta hai, sirf exact words nahi

        **Fine-tuned on:** 1 billion+ sentence pairs for semantic similarity tasks
        """)


# ══════════════════════════════════════════════════════════════
# PAGE: EXPLORE DATA
# ══════════════════════════════════════════════════════════════
elif page == "📊 Explore Data":
    st.markdown('<div class="main-header">📊 Dataset Explorer</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Ratings", "🎭 Genres", "📅 Timeline", "🔎 Search"])

    with tab1:
        bins  = [0,1,2,3,4,5,6,7,8,9,10]
        lbls  = [f"{b}-{b+1}" for b in bins[:-1]]
        valid = df['vote_average'].dropna()
        cnts  = pd.cut(valid, bins=bins, labels=lbls).value_counts().sort_index()
        st.markdown("### Rating Distribution")
        st.bar_chart(pd.DataFrame({'Count': cnts.values}, index=cnts.index))
        v = df['vote_average'].dropna().values
        c1,c2,c3 = st.columns(3)
        c1.metric("Mean",    f"{float(np.mean(v)):.2f}")
        c2.metric("Median",  f"{float(np.median(v)):.2f}")
        c3.metric("8+ rated",f"{int((v>=8).sum()):,}")

    with tab2:
        from collections import Counter
        gc = Counter(g for row in df['genre_str'].dropna() for g in row.split() if g)
        st.markdown("### Top Genres by Count")
        st.bar_chart(pd.DataFrame(gc.most_common(15), columns=['Genre','Count']).set_index('Genre'))
        st.markdown("### Avg Rating by Genre")
        gr = {g: round(float(np.mean(df[df['genre_str'].str.contains(g,na=False,case=False)]['vote_average'].dropna().values)),2)
              for g in ALL_GENRES[:15]
              if len(df[df['genre_str'].str.contains(g,na=False,case=False)]) > 50}
        gr_df = pd.DataFrame(list(gr.items()), columns=['Genre','Avg']).sort_values('Avg',ascending=False).set_index('Genre')
        st.bar_chart(gr_df)

    with tab3:
        st.markdown("### Movies Per Year (1970-2017)")
        yf = df[(df['year']>=1970)&(df['year']<=2017)]
        st.line_chart(yf['year'].value_counts().sort_index())

    with tab4:
        q = st.text_input("Search by title:")
        if q:
            st.dataframe(
                df[df['title'].str.contains(q,case=False,na=False)][
                    ['title','year','vote_average','vote_count','genre_str']].head(20),
                use_container_width=True)