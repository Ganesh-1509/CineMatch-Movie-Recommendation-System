"""CineMatch: content-based movie recommendations (Streamlit + scikit-learn + TMDB)."""

import ast
import html
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("dataset/tmdb_5000_movies.csv")

TMDB_API = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

try:
    TMDB_API_KEY = str(st.secrets["TMDB_API_KEY"]).strip()
except (KeyError, FileNotFoundError):
    TMDB_API_KEY = None  # app still works, posters show a placeholder

TOP_N = 10
MIN_CONTENT_SIMILARITY = 0.08
MIN_VOTES = 100
MIN_RATING = 5.0
VOTE_CONFIDENCE_K = 100  # votes needed before a rating is trusted ~50%

# One entry per mode: weights inside content similarity (genre/keyword/overview)
# and weights of the final score (content/popularity/quality).
MODES = {
    "⚖️ Balanced": dict(genre=0.25, keyword=0.40, overview=0.35,
                        content=0.75, popularity=0.15, quality=0.10),
    "📖 Story Focus": dict(genre=0.10, keyword=0.45, overview=0.45,
                           content=0.85, popularity=0.10, quality=0.05),
    "🎭 Genre Focus": dict(genre=0.65, keyword=0.20, overview=0.15,
                           content=0.85, popularity=0.10, quality=0.05),
    "🔥 Popular Movies": dict(genre=0.25, keyword=0.35, overview=0.40,
                              content=0.55, popularity=0.35, quality=0.10),
}
DEFAULT_MODE = "⚖️ Balanced"


# ============================================================
# DATA
# ============================================================

def _extract_names(text):
    """Parse a TMDB JSON-ish string like "[{'id': 28, 'name': 'Action'}]"."""
    try:
        return [item["name"] for item in ast.literal_eval(text)]
    except Exception:
        return []


@st.cache_data
def load_data():
    movies = pd.read_csv(DATA_PATH).reset_index(drop=True)

    for col in ("title", "overview", "keywords", "genres"):
        movies[col] = movies[col].fillna("")

    genre_lists = movies["genres"].apply(_extract_names)
    movies["genre_list"] = genre_lists
    movies["genres"] = genre_lists.apply(", ".join)  # for display
    # "Science Fiction" -> "ScienceFiction" so it stays ONE token for TF-IDF
    movies["genre_tokens"] = genre_lists.apply(
        lambda names: " ".join(n.replace(" ", "") for n in names)
    )
    movies["keywords"] = movies["keywords"].apply(_extract_names).apply(" ".join)

    movies["release_year"] = pd.to_datetime(
        movies["release_date"], errors="coerce"
    ).dt.year

    # Popularity is heavily skewed, so use a log scale (linear /max flattens
    # almost every movie to ~0).
    popularity = movies["popularity"].fillna(0).clip(lower=0)
    movies["popularity"] = popularity
    peak = np.log1p(popularity.max())
    movies["popularity_score"] = np.log1p(popularity) / peak if peak > 0 else 0.0

    # Rating shrunk by vote count so 9.0 from 10 votes doesn't beat 8.0 from 10k.
    votes = movies["vote_count"].fillna(0)
    rating = movies["vote_average"].fillna(0)
    movies["vote_count"], movies["vote_average"] = votes, rating
    movies["quality_score"] = (rating / 10) * (votes / (votes + VOTE_CONFIDENCE_K))

    return movies


try:
    tmdb_movies = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`.")
    st.stop()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def create_model(_movies):
    """Sparse TF-IDF matrices only. Similarity is computed per query, which
    avoids three dense 4803x4803 matrices (~550 MB)."""

    def fit(column, **kwargs):
        return TfidfVectorizer(dtype=np.float32, **kwargs).fit_transform(_movies[column])

    return {
        "genre": fit("genre_tokens"),
        "keyword": fit("keywords", stop_words="english"),
        "overview": fit("overview", stop_words="english", ngram_range=(1, 2)),
    }


tfidf = create_model(tmdb_movies)


# ============================================================
# POSTERS
# ============================================================

@st.cache_resource
def poster_store():
    """Shared dict: movie id -> poster URL (or None if TMDB has no poster).
    Failed requests are never stored, so they get retried next time."""
    return {}


@st.cache_resource
def http_session():
    """One shared session with automatic retries/backoff for flaky networks."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.7, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    return session


def _tmdb_get(path, params=None):
    """GET from TMDB. Works with both a v3 API key and a v4 read-access token."""
    params = dict(params or {})
    headers = {}
    if TMDB_API_KEY.startswith("eyJ"):  # v4 token (JWT)
        headers["Authorization"] = f"Bearer {TMDB_API_KEY}"
    else:
        params["api_key"] = TMDB_API_KEY
    return http_session().get(
        f"{TMDB_API}{path}", params=params, headers=headers, timeout=(5, 12)
    )


def _fetch_poster(movie_id, title, year):
    """Poster URL, or None if TMDB genuinely has none.
    Raises on network/API trouble so the failure is not cached."""
    response = _tmdb_get(f"/movie/{movie_id}")
    if response.status_code == 200:
        path = response.json().get("poster_path")
        if path:
            return TMDB_IMAGE_BASE + path
    elif response.status_code != 404:
        response.raise_for_status()

    # Fallback: title search (with release year to avoid wrong matches)
    params = {"query": title}
    if pd.notna(year):
        params["year"] = int(year)
    response = _tmdb_get("/search/movie", params)
    response.raise_for_status()
    results = response.json().get("results", [])
    if results and results[0].get("poster_path"):
        return TMDB_IMAGE_BASE + results[0]["poster_path"]
    return None


def get_posters(recommendations):
    """Fetch missing posters (a few in parallel, then one sequential second
    chance for failures). Returns ({movie_id: url_or_None}, {movie_id: error})."""
    store = poster_store()
    errors = {}
    ids = [int(i) for i in recommendations["id"]]

    if TMDB_API_KEY:
        wanted = [r for r in recommendations.itertuples() if int(r.id) not in store]

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                (r, pool.submit(_fetch_poster, int(r.id), r.title, r.release_year))
                for r in wanted
            ]
            for r, future in futures:
                try:
                    store[int(r.id)] = future.result()
                except Exception as exc:
                    errors[int(r.id)] = f"{type(exc).__name__}: {exc}"

        for r in wanted:  # second chance, one at a time
            if int(r.id) in errors:
                try:
                    store[int(r.id)] = _fetch_poster(int(r.id), r.title, r.release_year)
                    del errors[int(r.id)]
                except Exception as exc:
                    errors[int(r.id)] = f"{type(exc).__name__}: {exc}"

    return {i: store.get(i) for i in ids}, errors


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def find_movie(query):
    """Row index of the best title match: exact first, then partial.
    When several titles match, the one with the most votes wins."""
    query = query.strip()
    titles = tmdb_movies["title"]

    matches = tmdb_movies[titles.str.lower() == query.lower()]
    if matches.empty:
        matches = tmdb_movies[
            titles.str.contains(query, case=False, regex=False, na=False)
        ]
    if matches.empty:
        return None
    return matches["vote_count"].idxmax()


def recommend_movies(movie_name, recommendation_mode=DEFAULT_MODE):
    """Returns (index_of_input_movie, top_recommendations_df),
    or (None, None) if the movie isn't found."""
    movie_index = find_movie(movie_name)
    if movie_index is None:
        return None, None

    w = MODES.get(recommendation_mode, MODES[DEFAULT_MODE])

    # TF-IDF rows are L2-normalised, so the dot product is the cosine similarity.
    content_similarity = sum(
        w[name] * linear_kernel(tfidf[name][movie_index], tfidf[name]).ravel()
        for name in ("genre", "keyword", "overview")
    )

    final_scores = (
        w["content"] * content_similarity
        + w["popularity"] * tmdb_movies["popularity_score"].to_numpy()
        + w["quality"] * tmdb_movies["quality_score"].to_numpy()
    )

    recs = (
        tmdb_movies.assign(
            similarity_score=final_scores,
            content_similarity=content_similarity,
        )
        .drop(index=movie_index)  # never recommend the input movie
    )

    recs = recs[
        (recs["content_similarity"] >= MIN_CONTENT_SIMILARITY)
        & (recs["vote_count"] >= MIN_VOTES)
        & (recs["vote_average"] >= MIN_RATING)
    ]
    recs = recs.sort_values("similarity_score", ascending=False).head(TOP_N)

    # Display score: relative to the best candidate for this search, so it is
    # not dragged down by the (naturally small) raw TF-IDF values.
    if not recs.empty:
        recs = recs.assign(
            match_pct=100 * recs["similarity_score"] / recs["similarity_score"].max()
        )
    return movie_index, recs


def explain_recommendation(original_movie, recommended_movie):
    explanations = []

    shared = set(original_movie["genre_list"]) & set(recommended_movie["genre_list"])
    if shared:
        explanations.append("Shared genres: " + ", ".join(sorted(shared)))
    if recommended_movie["content_similarity"] >= 0.20:
        explanations.append("Similar storyline and content")
    if recommended_movie["vote_average"] >= 7.5:
        explanations.append("Highly rated by viewers")
    if recommended_movie["popularity"] >= 100:
        explanations.append("Popular among movie audiences")

    return explanations or ["Recommended based on similarities"]


# ============================================================
# PAGE STYLES
# ============================================================

PAGE_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] { background:#080A0B !important; color:#E4E9E9 !important; }
[data-testid="stHeader"] { background:rgba(8,10,11,0.95) !important; }
.block-container { max-width:1450px; padding-top:2rem; padding-bottom:4rem; }

.stMarkdown p, label { color:#D4DCDC; }
h1, h2, h3, h4 { color:#E8EEEE !important; font-weight:500 !important; }
.main-title { font-size:65px; font-weight:800; letter-spacing:-4px; color:#E8EEEE; }
.main-title span { color:#78C5C2; }
.subtitle { color:#718181; font-size:12px; letter-spacing:3px; text-transform:uppercase; margin-bottom:30px; }
.hint { color:#758585; font-size:12px; letter-spacing:1px; margin-top:-10px; }

[data-testid="stForm"] { border:none !important; padding:0 !important; }

.stTextInput label, .stSelectbox label { color:#A4B1B1 !important; }
.stTextInput input {
    background-color:#111617 !important; color:#E5EEEE !important;
    border:1px solid #293536 !important; border-radius:5px !important; height:52px !important;
}
.stTextInput input::placeholder { color:#647272 !important; }
.stTextInput input:focus { border:1px solid #78C5C2 !important; }

.stSelectbox div[data-baseweb="select"] > div {
    background-color:#111617 !important; color:#E5EEEE !important;
    border:1px solid #293536 !important; border-radius:5px !important;
}
.stSelectbox div[data-baseweb="select"] span { color:#D4DCDC !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color:#0C1011 !important; }
li[role="option"] { background-color:#0C1011 !important; color:#D4DCDC !important; }
li[role="option"]:hover { background-color:#182223 !important; }

.stButton button, [data-testid="stFormSubmitButton"] button {
    background-color:#182324 !important; color:#78C5C2 !important;
    border:1px solid #426263 !important; border-radius:5px !important;
    height:52px !important; width:100% !important;
    font-weight:600 !important; letter-spacing:1px !important; transition:all .25s ease;
}
.stButton button p, [data-testid="stFormSubmitButton"] button p { color:#78C5C2 !important; }
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
    background-color:#243738 !important; border:1px solid #78C5C2 !important;
}

div[data-testid="stAlert"] { background-color:#111617 !important; color:#D4DCDC !important; border:1px solid #293536 !important; }
hr { border-color:#263334 !important; }

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#080A0B; }
::-webkit-scrollbar-thumb { background:#304647; border-radius:10px; }

.footer { color:#526162; text-align:center; letter-spacing:3px; font-size:10px; padding-top:40px; line-height:1.8; }
</style>
"""

st.markdown(PAGE_CSS, unsafe_allow_html=True)


# ============================================================
# GALLERY (rendered in an iframe; __CARDS__ is replaced with HTML)
# ============================================================

GALLERY_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* { box-sizing:border-box; }
html, body { margin:0; padding:0; width:100%; background:transparent; font-family:Arial,Helvetica,sans-serif; color:#E4E9E9; overflow-x:hidden; }
body { padding:0 5px; }

/* Padding leaves room for the hovered card to scale up without being clipped */
.movie-gallery {
    display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:24px;
    padding:70px 55px 90px; position:relative; isolation:isolate;
}

.movie-card {
    position:relative; width:100%; min-width:0; aspect-ratio:2/3; overflow:hidden;
    border-radius:9px; background:#101516; border:1px solid #263334;
    transform-origin:center center; outline:none; cursor:pointer;
    transition:transform .35s ease, opacity .35s ease, box-shadow .35s ease, border-color .35s ease;
}
.movie-card:focus-visible { border-color:#78C5C2; }

.poster-container { position:absolute; inset:0; }
.poster-image { position:absolute; inset:0; z-index:1; display:block; width:100%; height:100%; object-fit:cover; }
.poster-placeholder {
    position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    background:#182223; color:#718181; text-align:center; padding:15px; font-size:13px;
}

.movie-basic-info { position:absolute; left:0; right:0; bottom:0; padding:12px; background:#101516; z-index:2; }
.movie-title { color:#E8EEEE; font-size:14px; font-weight:700; line-height:1.4; min-height:38px; margin-bottom:8px; overflow-wrap:anywhere; }
.movie-match { color:#78C5C2; font-size:12px; font-weight:700; }

/* Hover overlay, confined to the card */
.movie-hover-content {
    position:absolute; inset:0; padding:15px; display:flex; flex-direction:column; overflow:hidden; z-index:5;
    background:linear-gradient(to bottom, rgba(0,0,0,.96) 0%, rgba(0,0,0,.90) 45%, rgba(0,0,0,.96) 100%);
    opacity:0; visibility:hidden; transition:opacity .25s ease, visibility .25s ease;
}
.movie-card.card-hovered .movie-hover-content { opacity:1; visibility:visible; }

.hover-scroll-area {
    width:100%; height:100%; min-height:0; overflow-y:auto; overflow-x:hidden;
    display:flex; flex-direction:column; overscroll-behavior:contain;
    scrollbar-width:thin; scrollbar-color:#426263 transparent;
}
.hover-scroll-area::-webkit-scrollbar { width:4px; }
.hover-scroll-area::-webkit-scrollbar-track { background:transparent; }
.hover-scroll-area::-webkit-scrollbar-thumb { background:#426263; border-radius:5px; }

.movie-hover-title { flex-shrink:0; color:#FFF; font-size:21px; font-weight:800; line-height:1.35; margin:0 0 12px; overflow-wrap:anywhere; }
.movie-hover-rating { flex-shrink:0; color:#78C5C2; font-size:13px; font-weight:600; line-height:1.5; margin-bottom:10px; }
.match-separator { color:#627777; }
.movie-hover-genres { flex-shrink:0; color:#C6D4D4; font-size:12px; line-height:1.55; margin-bottom:12px; overflow-wrap:anywhere; }
.movie-overview { color:#E0E5E5; font-size:12px; line-height:1.65; margin:0 0 15px; overflow-wrap:anywhere; }
.why-title { flex-shrink:0; color:#78C5C2; font-size:13px; font-weight:800; letter-spacing:1px; line-height:1.5; margin:0 0 8px; }
.why-content { color:#DDE5E5; font-size:12px; line-height:1.6; }
.explanation-item { margin-bottom:7px; overflow-wrap:anywhere; }

/* Dim the others only while a card is active */
.movie-gallery.card-hover-active .movie-card:not(.card-hovered) { opacity:.25; transform:scale(.82); }
.movie-gallery.card-hover-active .movie-card.card-hovered {
    opacity:1; transform:scale(1.32); z-index:100; border-color:#78C5C2;
    box-shadow:0 20px 55px rgba(0,0,0,.95), 0 0 12px rgba(120,197,194,.12);
}

@media (max-width:1100px) {
    .movie-gallery { grid-template-columns:repeat(3,minmax(0,1fr)); gap:20px; padding:60px 40px 80px; }
    .movie-gallery.card-hover-active .movie-card.card-hovered { transform:scale(1.22); }
    .movie-gallery.card-hover-active .movie-card:not(.card-hovered) { transform:scale(.90); }
}

@media (max-width:700px) {
    .movie-gallery { grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; padding:25px 10px 40px; }
    .movie-card { border-radius:7px; }
    .movie-basic-info { padding:8px; }
    .movie-title { font-size:12px; min-height:32px; }
    .movie-match { font-size:10px; }
    .movie-hover-content { padding:10px; }
    .movie-hover-title { font-size:15px; line-height:1.3; margin-bottom:8px; }
    .movie-hover-rating { font-size:11px; margin-bottom:7px; }
    .movie-hover-genres { font-size:10px; margin-bottom:7px; }
    .movie-overview { font-size:10px; line-height:1.5; margin-bottom:10px; }
    .why-title { font-size:11px; }
    .why-content { font-size:10px; line-height:1.5; }
    .movie-gallery.card-hover-active .movie-card.card-hovered { transform:scale(1.06); z-index:100; opacity:1; }
    .movie-gallery.card-hover-active .movie-card:not(.card-hovered) { transform:scale(.94); opacity:.30; }
}

@media (max-width:400px) {
    .movie-gallery { gap:10px; }
    .movie-hover-content { padding:8px; }
    .movie-hover-title { font-size:13px; }
    .movie-hover-rating, .movie-hover-genres, .movie-overview, .why-content { font-size:9px; }
}
</style>
</head>
<body>

<div class="movie-gallery" id="movieGallery">
__CARDS__
</div>

<script>
const gallery = document.getElementById("movieGallery");
const cards = Array.from(gallery.querySelectorAll(".movie-card"));
const mouseLike = window.matchMedia("(hover: hover) and (pointer: fine)");
const touchLike = window.matchMedia("(hover: none) and (pointer: coarse)");
let selectedCard = null;

function activate(card) {
    cards.forEach((c) => c.classList.toggle("card-hovered", c === card));
    gallery.classList.add("card-hover-active");
    selectedCard = card;
}

function deactivate() {
    cards.forEach((c) => c.classList.remove("card-hovered"));
    gallery.classList.remove("card-hover-active");
    selectedCard = null;
}

cards.forEach((card) => {
    // Desktop: hover (and keyboard focus)
    card.addEventListener("mouseenter", () => { if (mouseLike.matches) activate(card); });
    card.addEventListener("mouseleave", () => { if (mouseLike.matches) deactivate(); });
    card.addEventListener("focus", () => { if (mouseLike.matches) activate(card); });
    card.addEventListener("blur", () => { if (mouseLike.matches) deactivate(); });

    // Mobile: tap to open, tap again to close
    card.addEventListener("click", (event) => {
        if (!touchLike.matches) return;
        event.stopPropagation();
        if (selectedCard === card) deactivate(); else activate(card);
    });
});

// Mobile: tap outside closes
document.addEventListener("click", (event) => {
    if (touchLike.matches && !event.target.closest(".movie-card")) deactivate();
});

// Size the iframe to its content (no nested scrollbars). If the browser
// blocks access to the frame, the iframe's own scrollbar is the fallback.
function fitFrame() {
    try {
        const height = Math.ceil(document.body.getBoundingClientRect().height);
        window.frameElement.style.height = height + "px";
        document.documentElement.style.overflowY = "hidden";
    } catch (e) {}
}
new ResizeObserver(fitFrame).observe(document.body);
window.addEventListener("load", fitFrame);
fitFrame();
</script>

</body>
</html>
"""


def build_card(movie, poster_url, explanations):
    title = html.escape(str(movie["title"]))
    overview = html.escape(str(movie["overview"]) or "No description available.")
    genres = html.escape(str(movie["genres"]) or "Genres unavailable")
    rating = float(movie["vote_average"])
    match = float(movie["match_pct"])

    # Placeholder always sits behind the image; if the image fails to load
    # in the browser it hides itself and the placeholder shows through.
    poster = '<div class="poster-placeholder">POSTER<br>UNAVAILABLE</div>'
    if poster_url:
        src = html.escape(poster_url, quote=True)
        poster += (
            f"""<img src="{src}" class="poster-image" alt="{title}" """
            """onerror="this.style.display='none'">"""
        )

    why = "".join(
        f"<div class='explanation-item'>• {html.escape(str(e))}</div>"
        for e in explanations
    )

    return f"""
    <div class="movie-card" tabindex="0">
        <div class="poster-container">{poster}</div>
        <div class="movie-basic-info">
            <div class="movie-title">{title}</div>
            <div class="movie-match">🎯 {match:.1f}% MATCH</div>
        </div>
        <div class="movie-hover-content">
            <div class="hover-scroll-area">
                <div class="movie-hover-title">{title}</div>
                <div class="movie-hover-rating">
                    ⭐ {rating:.1f}/10
                    <span class="match-separator">|</span>
                    🎯 {match:.1f}% Match
                </div>
                <div class="movie-hover-genres">🎭 {genres}</div>
                <div class="movie-overview">{overview}</div>
                <div class="why-title">WHY THIS FILM?</div>
                <div class="why-content">{why}</div>
            </div>
        </div>
    </div>
    """


def render_results(result):
    movie_index, recommendations = result["index"], result["recommendations"]

    if movie_index is None:
        st.error("Movie not found. Please try another title.")
        return
    if recommendations.empty:
        st.warning("No similar movies found.")
        return

    original_movie = tmdb_movies.loc[movie_index]

    st.success(
        f"Recommendations based on **{original_movie['title']}** "
        f"using **{result['mode']}** mode."
    )
    st.markdown("## 🍿 Your Recommendations")
    st.markdown(
        '<div class="hint">HOVER OVER OR TAP A POSTER TO DISCOVER MORE</div>',
        unsafe_allow_html=True,
    )

    missing_posters = [i for i, url in result["posters"].items() if not url]
    if missing_posters and result.get("poster_errors"):
        first_error = next(iter(result["poster_errors"].values()))
        st.caption(f"⚠️ {len(result['poster_errors'])} poster(s) couldn't be loaded ({first_error})")
        if st.button("🔄 Retry missing posters"):
            with st.spinner("Fetching posters..."):
                posters, errors = get_posters(recommendations)
            result["posters"], result["poster_errors"] = posters, errors
            st.rerun()

    cards = "".join(
        build_card(
            movie,
            result["posters"].get(int(movie["id"])),
            explain_recommendation(original_movie, movie),
        )
        for _, movie in recommendations.iterrows()
    )

    components.html(
        GALLERY_TEMPLATE.replace("__CARDS__", cards),
        height=1100,  # initial height; the page script resizes it to fit
        scrolling=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Cine<span>Match</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Discover your next favourite movie with Machine Learning</div>',
    unsafe_allow_html=True,
)
st.divider()

if TMDB_API_KEY is None:
    st.warning("TMDB_API_KEY is not set in secrets, so posters will be unavailable.")


# ============================================================
# SEARCH (a form: Enter submits, and results survive widget changes)
# ============================================================

with st.form("search_form"):
    col1, col2 = st.columns([2, 1])

    with col1:
        movie_name = st.text_input(
            "🎥 Search for a movie",
            placeholder="Try Interstellar, Cars, Toy Story...",
        )
    with col2:
        recommendation_mode = st.selectbox("🎛️ Recommendation Mode", list(MODES))

    submitted = st.form_submit_button("✨ Find My Movies", type="primary")

if submitted:
    query = movie_name.strip()
    if not query:
        st.warning("Please enter a movie name.")
    else:
        with st.spinner("🎬 Searching for movies you'll love..."):
            movie_index, recommendations = recommend_movies(query, recommendation_mode)
            posters, poster_errors = {}, {}
            if recommendations is not None and not recommendations.empty:
                posters, poster_errors = get_posters(recommendations)

        st.session_state["result"] = {
            "index": movie_index,
            "recommendations": recommendations,
            "mode": recommendation_mode,
            "posters": posters,
            "poster_errors": poster_errors,
        }

if "result" in st.session_state:
    render_results(st.session_state["result"])


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        BUILT WITH PYTHON · STREAMLIT · SCIKIT-LEARN · TMDB<br>
        This product uses the TMDB API but is not endorsed or certified by TMDB.
    </div>
    """,
    unsafe_allow_html=True,
)