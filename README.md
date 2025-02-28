# Personalized Movie Recommendation System

A Python-based movie recommendation system that uses collaborative filtering to suggest films based on user preferences. Featuring a natural-language chatbot interface, this project integrates a SQLite database to track viewed movies and ratings, evaluates performance with precision and recall, and refines suggestions through user feedback. Built with data inspired by Rotten Tomatoes reviews, it’s designed to make movie nights more enjoyable with tailored picks.

---

## Features

- **Personalized Recommendations**: Uses the SVD algorithm from Surprise to recommend movies based on user ratings and viewing history.
- **Natural Chatbot Interface**: Conversational input parsing (e.g., "I’ve seen Inception", "Recommend something") with fuzzy title matching for flexibility.
- **User Interaction**:
  - Mark movies as seen.
  - Rate movies (1-5).
  - Request recommendations and provide feedback ("yes/no") to enhance the model.
  - View a list of seen/rated movies.
- **Persistent Storage**: SQLite database stores user ratings and seen movies.
- **Evaluation Metrics**: Offline evaluation with RMSE, Precision@10, and Recall@10 to assess recommendation quality.

---

## Tech Stack

- **Python 3.9+**: Core programming language.
- **Libraries**:
  - **Pandas**: Data manipulation and preprocessing.
  - **Surprise**: Collaborative filtering with SVD.
  - **FuzzyWuzzy**: Fuzzy string matching for movie titles.
  - **SQLAlchemy**: SQLite database integration.
- **Dataset**: Hypothetical "Massive Rotten Tomatoes Movies & Reviews" (Kaggle, 1.4M+ reviews, 140K+ movies).

---

## Installation

### Prerequisites

- Python 3.9 or higher.
- Git (optional, for cloning).
- A downloaded dataset (see below).

### Steps

1. **Clone the Repository** (or download the ZIP):
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
