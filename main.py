import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from fuzzywuzzy import process
import sqlite3
import re
import os
from sqlalchemy import create_engine

# --- Helper Functions ---

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Compute precision and recall at k for evaluating recommendations."""
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    
    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    return precision, recall

def find_movie(title, movies_df):
    """Find a movie ID by fuzzy matching the title."""
    matches = process.extract(title, movies_df['title'], limit=5)
    if matches[0][1] >= 90:
        return movies_df[movies_df['title'] == matches[0][0]]['movie_id'].values[0]
    else:
        print("Did you mean one of these?")
        for i, match in enumerate(matches):
            print(f"{i+1}. {match[0]} (Similarity: {match[1]}%)")
        choice = input("Enter the number or 'none': ").strip()
        if choice.isdigit() and 1 <= int(choice) <= 5:
            return movies_df[movies_df['title'] == matches[int(choice)-1][0]]['movie_id'].values[0]
        return None

def initialize_database():
    """Set up SQLite database for user interactions."""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_ratings
                 (user_id INTEGER, movie_id INTEGER, rating REAL, seen INTEGER,
                  PRIMARY KEY (user_id, movie_id))''')
    conn.commit()
    conn.close()

def load_user_ratings(user_id):
    """Load user's ratings and seen movies from the database."""
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT * FROM user_ratings WHERE user_id = ?", conn, params=(user_id,))
    conn.close()
    return df

def save_user_rating(user_id, movie_id, rating=None, seen=1):
    """Save or update a user's movie rating and seen status."""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    # Check if entry exists
    c.execute("SELECT rating, seen FROM user_ratings WHERE user_id = ? AND movie_id = ?",
              (user_id, movie_id))
    result = c.fetchone()
    if result:
        # Update existing entry
        current_rating, current_seen = result
        new_rating = rating if rating is not None else current_rating
        new_seen = 1  # Seen is always 1 if interacted with
        c.execute("UPDATE user_ratings SET rating = ?, seen = ? WHERE user_id = ? AND movie_id = ?",
                  (new_rating, new_seen, user_id, movie_id))
    else:
        # Insert new entry
        rating = rating if rating is not None else None
        c.execute("INSERT INTO user_ratings (user_id, movie_id, rating, seen) VALUES (?, ?, ?, ?)",
                  (user_id, movie_id, rating, seen))
    conn.commit()
    conn.close()

def get_recommendations(user_id, ratings_df, movies_df, n=5):
    """Generate personalized movie recommendations."""
    user_ratings = load_user_ratings(user_id)
    all_ratings = pd.concat([ratings_df, user_ratings[['user_id', 'movie_id', 'rating']].dropna()])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(all_ratings[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    
    seen_movies = user_ratings['movie_id'].unique()
    all_movies = movies_df['movie_id'].unique()
    unseen_movies = [m for m in all_movies if m not in seen_movies]
    
    if not unseen_movies:
        return ["Looks like you've seen everything! Try rating more movies."]
    
    predictions = [algo.predict(str(user_id), str(m)) for m in unseen_movies]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return [movies_df[movies_df['movie_id'] == int(p.iid)]['title'].values[0] for p in top_n]

# --- Main Execution ---

# Load Rotten Tomatoes dataset (assumed structure)
print("Loading Rotten Tomatoes dataset...")
ratings_df = pd.read_csv('rotten_tomatoes_reviews.csv')  # Adjust path if needed
# Assuming columns: user_id, movie_id, title, score (rating)
ratings_df = ratings_df.rename(columns={'score': 'rating'})
ratings_df = ratings_df[['user_id', 'movie_id', 'rating']].dropna()
ratings_df['user_id'] = ratings_df['user_id'].astype(str)
ratings_df['movie_id'] = ratings_df['movie_id'].astype(int)
ratings_df['rating'] = ratings_df['rating'].astype(float)

# Load or create movies DataFrame (assuming titles are in ratings or separate file)
if 'title' in ratings_df.columns:
    movies_df = ratings_df[['movie_id', 'title']].drop_duplicates()
else:
    movies_df = pd.read_csv('movies.csv')  # Adjust if you have a separate movies file
movies_df['movie_id'] = movies_df['movie_id'].astype(int)

# Offline Evaluation
print("Performing offline evaluation...")
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], Reader(rating_scale=(1, 5)))
trainset, testset = train_test_split(data, test_size=0.2)
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
rmse_score = rmse(predictions)
precision, recall = precision_recall_at_k(predictions, k=10, threshold=3.5)
print(f"Offline Evaluation Results:")
print(f"RMSE: {rmse_score:.4f}")
print(f"Precision@10: {precision:.4f}")
print(f"Recall@10: {recall:.4f}")

# Initialize database
initialize_database()

# Assign user ID (could be dynamic for multiple users)
user_id = ratings_df['user_id'].astype(int).max() + 1 if ratings_df['user_id'].str.match(r'^\d+$').all() else 1000

# --- Chatbot Interface ---

print("Hey there! I’m your movie guru. I can recommend films, track what you’ve seen, and learn from your tastes.")
print("Try saying things like:")
print("- 'I’ve seen Inception'")
print("- 'Rate The Matrix 4'")
print("- 'Recommend me something'")
print("- 'Show my movies'")
print("- Or 'exit' to quit")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Catch you later, movie buff!")
        break
    
    # Parse input with regex for natural language
    seen_match = re.search(r"(?:i(?:'ve)?\s*(?:watched|seen)\s+)(.+)", user_input, re.IGNORECASE)
    rate_match = re.search(r"(?:rate\s+)(.+?)\s+(\d\.?\d*)", user_input, re.IGNORECASE)
    recommend_match = re.search(r"(?:recommend|suggest)(?:\s+me)?\s*(something|a movie)?", user_input, re.IGNORECASE)
    show_match = re.search(r"(?:show|list)\s*(?:my)?\s*(movies|seen|ratings)", user_input, re.IGNORECASE)
    
    if seen_match:
        title = seen_match.group(1).strip()
        movie_id = find_movie(title, movies_df)
        if movie_id:
            save_user_rating(user_id, movie_id, seen=1)
            print(f"Noted! You’ve seen '{movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]}'.")
            rating = input("Want to rate it (1-5)? Say 'no' to skip: ").strip()
            if rating.lower() != 'no':
                try:
                    rating = float(rating)
                    if 1 <= rating <= 5:
                        save_user_rating(user_id, movie_id, rating=rating)
                        print(f"Awesome, rated as {rating}!")
                    else:
                        print("Oops, rating should be between 1 and 5.")
                except ValueError:
                    print("Hmm, that’s not a valid rating.")
        else:
            print("Couldn’t find that movie. Try again?")
    
    elif rate_match:
        title, rating_str = rate_match.groups()
        try:
            rating = float(rating_str)
            if 1 <= rating <= 5:
                movie_id = find_movie(title, movies_df)
                if movie_id:
                    save_user_rating(user_id, movie_id, rating=rating)
                    print(f"You’ve rated '{movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]}' as {rating}. Thanks!")
                else:
                    print("Movie not found. Check the title?")
            else:
                print("Rating should be between 1 and 5.")
        except ValueError:
            print("That rating doesn’t look right. Use a number like '4' or '3.5'.")
    
    elif recommend_match:
        user_ratings = load_user_ratings(user_id)
        if len(user_ratings[user_ratings['rating'].notna()]) < 3:
            print("I need a few more ratings to tailor recommendations. Rate some movies first!")
        else:
            recs = get_recommendations(user_id, ratings_df, movies_df)
            print("Here’s what I think you’ll enjoy:")
            for i, title in enumerate(recs, 1):
                print(f"{i}. {title}")
            feedback = input("Did you like these suggestions? (yes/no/for each #: yes/no): ").strip().lower()
            if feedback in ['yes', 'no']:
                # Simplified feedback: apply to all
                for i, title in enumerate(recs, 1):
                    movie_id = movies_df[movies_df['title'] == title]['movie_id'].values[0]
                    rating = 4 if feedback == 'yes' else 2
                    save_user_rating(user_id, movie_id, rating=rating)
                print(f"Feedback noted! I’ll tweak my suggestions.")
            elif feedback:
                # Per-movie feedback
                feedbacks = feedback.split()
                for i, fb in enumerate(feedbacks, 1):
                    if i <= len(recs) and fb in ['yes', 'no']:
                        movie_id = movies_df[movies_df['title'] == recs[i-1]]['movie_id'].values[0]
                        rating = 4 if fb == 'yes' else 2
                        save_user_rating(user_id, movie_id, rating=rating)
                print("Got it, I’ll refine my picks!")
    
    elif show_match:
        user_ratings = load_user_ratings(user_id)
        if user_ratings.empty:
            print("You haven’t logged any movies yet. Start by saying 'I’ve seen [movie]'!")
        else:
            print("Here’s what you’ve logged:")
            for _, row in user_ratings.iterrows():
                title = movies_df[movies_df['movie_id'] == row['movie_id']]['title'].values[0]
                rating = f" - Rated: {row['rating']}" if pd.notna(row['rating']) else ""
                print(f"- {title}{rating}")
    
    else:
        print("Not sure what you mean. Try 'I’ve seen [movie]', 'rate [movie] [rating]', 'recommend', or 'show my movies'.")