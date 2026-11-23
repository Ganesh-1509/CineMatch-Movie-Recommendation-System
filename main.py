import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# 1. LOAD DATASET
# ==============================

tmdb_movies = pd.read_csv("dataset/tmdb_5000_movies.csv")

print(tmdb_movies[["title", "popularity", "vote_average", "vote_count"]].head())
# ==============================
# 2. HANDLE MISSING VALUES
# ==============================

tmdb_movies["overview"] = tmdb_movies["overview"].fillna("")
tmdb_movies["keywords"] = tmdb_movies["keywords"].fillna("")
tmdb_movies["genres"] = tmdb_movies["genres"].fillna("")


# ==============================
# 3. EXTRACT GENRES AND KEYWORDS
# ==============================

def extract_names(text):
    try:
        data = ast.literal_eval(text)
        return " ".join(item["name"] for item in data)
    except:
        return ""


tmdb_movies["genres"] = tmdb_movies["genres"].apply(extract_names)
tmdb_movies["keywords"] = tmdb_movies["keywords"].apply(extract_names)


# ==============================
# 4. COMBINE MOVIE FEATURES
# ==============================

tmdb_movies["combined_features"] = (
    tmdb_movies["genres"] + " " +
    tmdb_movies["keywords"] + " " +
    tmdb_movies["overview"]
)


# ==============================
# 5. TF-IDF VECTORIZATION
# ==============================

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    tmdb_movies["combined_features"]
)


# ==============================
# 6. CALCULATE SIMILARITY
# ==============================

similarity_matrix = cosine_similarity(tfidf_matrix)
# Normalize popularity scores
tmdb_movies["popularity_normalized"] = (
    tmdb_movies["popularity"] /
    tmdb_movies["popularity"].max()
)

# Normalize ratings
tmdb_movies["rating_normalized"] = (
    tmdb_movies["vote_average"] / 10
)


# ==============================
# 7. RECOMMENDATION FUNCTION
# ==============================

def recommend_movies(movie_name):

    # Clean user input
    movie_name = movie_name.strip().lower()

    # Clean movie titles
    movie_titles = (
        tmdb_movies["title"]
        .fillna("")
        .str.strip()
        .str.lower()
    )

    # Search for exact match
    exact_matches = tmdb_movies[
        movie_titles == movie_name
    ]

    if not exact_matches.empty:
        matches = exact_matches

    else:
        # Search for partial match
        matches = tmdb_movies[
            movie_titles.str.contains(
                movie_name,
                na=False,
                regex=False
            )
        ]

    # Movie not found
    if matches.empty:
        print("\nMovie not found!")
        return

    # Handle multiple matches
    if len(matches) > 1:

        print("\nMultiple movies found:\n")

        displayed_matches = matches["title"].tolist()[:10]

        for i, title in enumerate(displayed_matches):
            print(f"{i + 1}. {title}")

        try:
            choice = int(input("\nEnter the movie number: "))

            if choice < 1 or choice > len(displayed_matches):
                print("Invalid selection!")
                return

            movie_index = matches.index[choice - 1]

        except ValueError:
            print("Please enter a valid number!")
            return

    else:
        movie_index = matches.index[0]

    selected_movie = tmdb_movies.loc[movie_index, "title"]

    # Get similarity scores
    similarity_scores = similarity_matrix[movie_index]

    # Calculate final recommendation score
    final_scores = (
        0.70 * similarity_scores +
        0.15 * tmdb_movies["popularity_normalized"].values +
        0.15 * tmdb_movies["rating_normalized"].values
    )

    # Sort movies by final score
    ranked_movies = list(enumerate(final_scores))

    ranked_movies = sorted(
        ranked_movies,
        key=lambda x: x[1],
        reverse=True
    )

    # Get top 10 recommendations
    recommendations = ranked_movies[1:11]

    # Display recommendations
    print(f"\nMovies similar to {selected_movie}:\n")

    for index, score in recommendations:
        print(
            tmdb_movies.loc[index, "title"],
            "| Score:",
            round(score, 3)
        )
# ==============================
# 8. USER INPUT
# ==============================

movie_name = input("Enter a movie name: ")

recommend_movies(movie_name)