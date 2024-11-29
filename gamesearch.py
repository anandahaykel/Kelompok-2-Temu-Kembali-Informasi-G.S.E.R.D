import pandas as pd
from collections import defaultdict
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # Changed to cosine similarity
from sklearn.preprocessing import normalize
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk


# --- Data Loading and Preprocessing ---

# Load the CSV file
file_path = 'games.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Convert all columns to string to prevent dtype issues
df = df.astype(str)

# Clean data: Replace NaN with empty strings
df.fillna('', inplace=True)

# Function to preprocess the platforms column
def preprocess_platforms(platform_data):
    try:
        platforms = ast.literal_eval(platform_data)
        return ', '.join(platform['platform']['name'] for platform in platforms)
    except (ValueError, SyntaxError):
        return ''
    except KeyError:
        print(f"Platform data structure is unexpected: {platform_data}")
        return ''

# Function to preprocess the tags column
def preprocess_tags(tag_data):
    try:
        return ', '.join(tag['name'] for tag in ast.literal_eval(tag_data))
    except (ValueError, SyntaxError):
        return ''

# Preprocess the platforms and tags columns
df['platforms'] = df['platforms'].apply(preprocess_platforms)
df['tags'] = df['tags'].apply(preprocess_tags)


# --- Text Preprocessing ---

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower() 

    words = nltk.word_tokenize(text)  # Tokenize the text 
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# --- Vector Space Model ---

# Combine relevant fields into a single text field for vectorization
df['text'] = df['name'].apply(preprocess_text) + ' ' + df['platforms'].apply(preprocess_text) + ' ' + df['tags'].apply(preprocess_text)

# Create TF-IDF vectors for each game
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Include unigrams and bigrams
tfidf_matrix = vectorizer.fit_transform(df['text'])

# Normalize the TF-IDF vectors
normalized_tfidf_matrix = normalize(tfidf_matrix)

# --- Search and Evaluation ---

def search_cosine_similarity(query):
    query_vector = vectorizer.transform([preprocess_text(query)])  # Preprocess the query
    # Normalize the query vector
    normalized_query_vector = normalize(query_vector)
    cosine_similarities = cosine_similarity(normalized_query_vector, normalized_tfidf_matrix).flatten()
    # Sort by similarity in descending order (higher similarity = more similar)
    sorted_indices = cosine_similarities.argsort()[::-1]
    # Return only the top 10 results with their distances
    return [(df['name'].iloc[i], cosine_similarities[i]) for i in sorted_indices[:10]]

def evaluate(results, expected_results):
    if not expected_results:
        return 0, 0
    true_positives = len(set(results) & set(expected_results))
    precision = true_positives / len(results) if results else 0
    recall = true_positives / len(expected_results)
    return precision, recall

def mean_average_precision(query, results):
    # If no results, no evaluation can be done
    if not results:
        return 0

    # Determine relevant documents based on the query
    relevant_documents = []
    query_lower = query.lower()
    for _, row in df.iterrows():  # Use _ for unused index
        # Check if ANY keywords in the query are present in the processed TEXT
        if any(keyword in row['text'] for keyword in query_lower.split()):
            relevant_documents.append(row['name'])  # Use the original game name

    if not relevant_documents:
        return 0  # No relevant documents for this query

    average_precisions = []
    relevant_count = 0
    for i, result in enumerate(results):
        if result in relevant_documents:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            average_precisions.append(precision_at_i)

    if not average_precisions:
        return 0  # No relevant documents found in the results

    mean_ap = sum(average_precisions) / len(average_precisions)
    return mean_ap

# --- Streamlit Web App ---

st.title("Game Search Engine")

query = st.text_input("Enter your search query:", "")

# Perform search automatically as the user types
results_with_distances = search_cosine_similarity(query)  # Call the correct search function

st.subheader("Search Results (Top 10):")
if results_with_distances:
    for i, (result, distance) in enumerate(results_with_distances):
        st.write(f"{i+1}. {result} (Distance: {distance:.4f})")  # Display rank and distance
else:
    st.write("No games found for your query.")

# --- Evaluation (Mean Average Precision) ---

# Extract just the game names from results_with_distances for evaluation
results = [result for result, distance in results_with_distances]  
st.subheader("Evaluation:")
mean_ap = mean_average_precision(query, results)
st.write(f"Mean Average Precision (MAP): {mean_ap:.2f}")