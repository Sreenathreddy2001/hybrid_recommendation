import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
# Sample dataset (Ensure your actual dataset is loaded from CSV)
data=pd.read_csv('/content/ratingS.csv')
# Convert to DataFrame
ratings_df = pd.DataFrame(data)
# Create user-book matrix
pivot_table = data.pivot_table(index="user_id", columns="book_id", values="rating", aggfunc='mean').fillna(0)
n_books = len(pivot_table.columns) # Get the number of unique books
# Apply Singular Value Decomposition (SVD) for Collaborative Filtering
svd = TruncatedSVD(n_components=n_books)
matrix = svd.fit_transform(pivot_table)
# Compute collaborative filtering similarity
collab_similarity = cosine_similarity(matrix.T)
# Hybrid Recommendation Function (Collaborative Filtering Only)
def hybrid_recommend(userid, num_recommendations=3):
    """ Recommend books for a user using collaborative filtering only """
    if userid not in pivot_table.index:
        return f"User {userid} not found in dataset!"
    # Get user preferences
    user_ratings = pivot_table.loc[userid]
    # Find books the user hasn't rated yet
    unrated_books = user_ratings[user_ratings == 0].index.tolist()
    recommendations = []
    for book in unrated_books:
        book_idx = list(pivot_table.columns).index(book)
        # Predict rating using collaborative filtering
        predicted_rating = np.mean(collab_similarity[book_idx])
        recommendations.append((book, userid, round(predicted_rating, 2)))
    # Sort by predicted rating
    recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)[:num_recommendations]
    # Convert to DataFrame with required column names
    return pd.DataFrame(recommendations, columns=["book_id", "user_id", "rating"])
# Example: Get recommendations for User 1169
print(hybrid_recommend(1169))
