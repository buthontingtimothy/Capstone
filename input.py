import numpy as np
import pandas as pd
# from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def generate_demographic_sentence(row):
    """
    Convert a user's demographic data into a long descriptive sentence.
    """
    # Fill missing values with placeholders
    sex = row['sex'] if pd.notna(row['sex']) else "a person"
    status = row['status'] if pd.notna(row['status']) else "an unspecified relationship status"
    orientation = row['orientation'] if pd.notna(row['orientation']) else "with an unspecified orientation"
    body_type = row['body_type'] if pd.notna(row['body_type']) else "an unspecified body type"
    education = row['education'] if pd.notna(row['education']) else "an unspecified education level"
    job = row['job'] if pd.notna(row['job']) else "no specific job mentioned"
    location = row['location'] if pd.notna(row['location']) else "an unspecified location"
    ethnicity = row['ethnicity'] if pd.notna(row['ethnicity']) else "an unspecified ethnicity"
    diet = row['diet'] if pd.notna(row['diet']) else "no specific diet preference"
    drinks = row['drinks'] if pd.notna(row['drinks']) else "an unspecified drinking habit"
    smokes = row['smokes'] if pd.notna(row['smokes']) else "an unspecified smoking habit"
    drugs = row['drugs'] if pd.notna(row['drugs']) else "an unspecified stance on drugs"
    pets = row['pets'] if pd.notna(row['pets']) else "no specific pet preference"
    religion = row['religion'] if pd.notna(row['religion']) else "no specific religion"
    sign = row['sign'] if pd.notna(row['sign']) else "no zodiac sign mentioned"
    speaks = row['speaks'] if pd.notna(row['speaks']) else "an unspecified language proficiency"

    # Construct the long descriptive sentence
    sentence = (f"{sex}, {status}, living in {location}, sexual orientation is {orientation}. "
                f"Has {body_type} body type and ethnicity is {ethnicity}. "
                f"Education level: {education}. industry: {job}. "
                f"Dietary preference: {diet}. Drinking habit: {drinks}. "
                f"Smoking habit: {smokes}. Drug use: {drugs}. "
                f"Pet preference: {pets}. Religion: {religion}. Zodiac sign: {sign}. "
                f"Speaks: {speaks}.")
    
    return sentence

# Generate demographic sentences for each user
def combine_demographics_essay(row):
    """
    Combine demographic data and essay data into a single sentence.
    """
    demographic_sentence = generate_demographic_sentence(row)
    essay = row['essay_all'] if pd.notna(row['essay_all']) else ""
    sentence = demographic_sentence + " " + essay
    return sentence


def sentence_embeddings(texts, model):
    """
    Generate sentence embeddings using a pre-trained Sentence Transformer model.
    """
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

def generate_embeddings(df, model):
    """
    Generate sentence embeddings for all user profiles in the dataset.
    """
    # Combine demographic data and essay data into a single sentence
    sentences = df.apply(combine_demographics_essay, axis=1).tolist()

    # Generate sentence embeddings
    embeddings = sentence_embeddings(sentences, model)
    
    return embeddings


def standardize_numeric(df):
    """
    Standardize non-text features in the dataset.
    """
    df = df.copy()

    for col in df.columns:
        if (df[col].dtype == 'float64') or (df[col].dtype == 'int64'):
            if len(df) != 1:
                df[col] = (df[col] - df[col].median()) / df[col].std()
            else:
                df_temp = pd.read_csv("okcupid_profiles_cleaned.csv")                
                df[col] = (df[col] - df_temp[col].median()) / df_temp[col].std()
    return df

def preprocess_data(df, model):
    """
    Preprocess the dataset by generating embeddings and standardizing non-text features.
    """
    # Generate embeddings
    embeddings = generate_embeddings(df, model)
    
    # Standardize non-text features
    non_text_features = df.select_dtypes(include=['float64', 'int64'])
    non_text_features = standardize_numeric(non_text_features)
    
    # Combine text and non-text features
    X = np.concatenate([embeddings, non_text_features], axis=1)

    return X


def input_embedding(input_dict, model):
    """
    Generate sentence embeddings for the user input data.
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Preprocess the input data
    embeddings = preprocess_data(input_df, model)

    return embeddings


def similarity_input(embeddings, embeddings_matrix):
    """
    Calculate the similarity between the user input and all profiles in the dataset.
    """
    # Calculate the cosine similarity between the input and all profiles
    similarities = np.dot(embeddings, embeddings_matrix.T)/(
        np.linalg.norm(embeddings, axis=1)*np.linalg.norm(embeddings_matrix.T))
    
    return similarities


input_dict = {'age': 22.0,
 'status': 'single',
 'sex': 'male',
 'orientation': 'straight',
 'body_type': 'a little extra',
 'diet': 'strictly anything',
 'drinks': 'socially',
 'drugs': 'never',
 'education': 'working on college/university',
 'ethnicity': 'asian, white',
 'height': 190.0,
 'job': 'transportation',
 'location': 'south san francisco, california',
 'offspring': "doesn't have kids, but might want them",
 'pets': 'likes dogs and likes cats',
 'religion': 'agnosticism and very serious about it',
 'sign': 'gemini',
 'smokes': 'sometimes',
 'speaks': 'english',
 'essay_all': "about me"}

embeddings_matrix = np.load("okcupid_profiles_preprocessed.npy")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = input_embedding(input_dict, sbert_model)

similarity = similarity_input(embeddings, embeddings_matrix[:2])
print(similarity.shape)

em_add = np.array([embeddings, embeddings_matrix[:2]])
em_add.shape

len([embeddings, embeddings_matrix[:2]])
cosine_similarity(embeddings)
