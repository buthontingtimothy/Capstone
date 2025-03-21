# Description: This file contains the functions to preprocess the input data and generate the top matches for the user input.

# data manipulation
import numpy as np
import pandas as pd
# get_top_matches function from find_match.py
from find_match import get_top_matches
# Sentence embeddings
from sentence_transformers import SentenceTransformer
import streamlit as st


def generate_demographic_sentence(row):
    """
    Convert a user's demographic data into a long descriptive sentence.
    """
    # Fill missing values with placeholders
    sex = row['sex'] if pd.notna(row['sex']) else "a person"
    status = row['status'] if pd.notna(
        row['status']) else "an unspecified relationship status"
    orientation = row['orientation'] if pd.notna(
        row['orientation']) else "with an unspecified orientation"
    body_type = row['body_type'] if pd.notna(
        row['body_type']) else "an unspecified body type"
    education = row['education'] if pd.notna(
        row['education']) else "an unspecified education level"
    job = row['job'] if pd.notna(row['job']) else "no specific job mentioned"
    location = row['location'] if pd.notna(
        row['location']) else "an unspecified location"
    ethnicity = row['ethnicity'] if pd.notna(
        row['ethnicity']) else "an unspecified ethnicity"
    diet = row['diet'] if pd.notna(
        row['diet']) else "no specific diet preference"
    drinks = row['drinks'] if pd.notna(
        row['drinks']) else "an unspecified drinking habit"
    smokes = row['smokes'] if pd.notna(
        row['smokes']) else "an unspecified smoking habit"
    drugs = row['drugs'] if pd.notna(
        row['drugs']) else "an unspecified stance on drugs"
    pets = row['pets'] if pd.notna(
        row['pets']) else "no specific pet preference"
    religion = row['religion'] if pd.notna(
        row['religion']) else "no specific religion"
    sign = row['sign'] if pd.notna(row['sign']) else "no zodiac sign mentioned"
    speaks = row['speaks'] if pd.notna(
        row['speaks']) else "an unspecified language proficiency"

    # Construct the long descriptive sentence
    sentence = (
        f"{sex}, {status}, living in {location}, sexual orientation is {orientation}. "
        f"Has {body_type} body type and ethnicity is {ethnicity}. "
        f"Education level: {education}. Industry: {job}. "
        f"Dietary preference: {diet}. Drinking habit: {drinks}. "
        f"Smoking habit: {smokes}. Drug use: {drugs}. "
        f"Pet preference: {pets}. Religion: {religion}. Zodiac sign: {sign}. "
        f"Speaks: {speaks}."
    )

    return sentence


def combine_demographics_essay(row):
    """
    Combine demographic data and essay data into a single sentence.
    """
    # Generate a long descriptive sentence for the demographic data
    demographic_sentence = generate_demographic_sentence(row)
    # Fill missing values with an empty string
    essay = row['essay_all'] if pd.notna(row['essay_all']) else ""
    # Combine demographic data and essay data into a single sentence
    sentence = demographic_sentence + " " + essay

    return sentence


def generate_embeddings(df, model):
    """
    Generate sentence embeddings for all user profiles in the dataset.
    """
    # Combine demographic data and essay data into a single sentence
    sentences = df.apply(combine_demographics_essay, axis=1).tolist()

    # Generate sentence embeddings
    embeddings = model.encode(sentences, show_progress_bar=False)

    return embeddings


def standardize_numeric(df, path_original_data):
    """
    Standardize non-text features in the dataset.
    """
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    df = df.copy()

    # Standardize numeric features
    for col in df.columns:
        # Standardize columns with numeric data types
        if (df[col].dtype == 'float64') or (df[col].dtype == 'int64'):
            # Standardize the column if there is more than one row in the DataFrame
            if len(df) != 1:
                # Standardize the column using the median and standard deviation of the column
                df[col] = (df[col] - df[col].median()) / df[col].std()
            else:
                # If there is only one row in the DataFrame, standardize using the median and standard deviation from the original dataset
                df_temp = pd.read_csv(path_original_data)
                # Standardize the column using the median and standard deviation from the original dataset
                df[col] = (df[col] - df_temp[col].median()) / \
                    df_temp[col].std()
    return df


def preprocess_data(df, model, path_original_data):
    """
    Preprocess the dataset by generating embeddings and standardizing non-text features.
    """
    # Generate embeddings
    embeddings = generate_embeddings(df, model)

    # Standardize non-text features
    non_text_features = df.select_dtypes(include=['float64', 'int64'])
    non_text_features_standardized = standardize_numeric(
        non_text_features, path_original_data)

    # Concatenate the embeddings and standardized non-text features
    X = np.concatenate([embeddings, non_text_features_standardized], axis=1)

    return X


def input_embedding(input_dict, model, path_original_data):
    """
    Generate sentence embeddings for the user input data.
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Preprocess the input data
    embeddings = preprocess_data(input_df, model, path_original_data)

    return embeddings


def similarity_input(embeddings, embeddings_matrix):
    """
    Calculate the similarity between the user input and all profiles in the dataset.
    """
    # Calculate the dot product between the user input and all profiles in the dataset
    dot_product = np.dot(embeddings, embeddings_matrix.T)  # Shape (2, 1)
    # Calculate the norm of the user input and all profiles in the dataset
    norm_array1 = np.linalg.norm(
        embeddings, axis=1, keepdims=True)  # Shape (2, 1)
    # Calculate the norm of the embeddings matrix
    norm_array2 = np.linalg.norm(
        embeddings_matrix, axis=1, keepdims=True)  # Shape (1, 1)
    # Calculate the cosine similarity
    similarities = dot_product / (norm_array1 * norm_array2.T)  # Shape (2, 1)

    return similarities

# "okcupid_profiles_preprocessed.npy" is the preprocessed embeddings matrix
# "all-MiniLM-L6-v2" is the SBERT model
# "okcupid_profiles_cleaned.csv" is the user data


def check_invalid_input(input_dict):
    """
    Check if the input dictionary contains any invalid values.
    """
    if not isinstance(input_dict['age'], (int, float)):
        return "Please input valid age"
    elif not (16 <= input_dict['age'] <= 120):
        return "Your age should be between 16 and 120"

    if (not isinstance(input_dict['height'], (int, float))):
        return "Please input valid height"
    elif not (50 <= input_dict['height'] <= 250):
        return "Your height should be between 50 and 250"

    if input_dict['sex'] not in ['male', 'female']:
        return "Please input valid sex"

    if input_dict['orientation'] not in ['straight', 'gay', 'bisexual']:
        return "Please input valid orientation"
    return None


@st.cache_data(show_spinner=False)
def load_embeddings_matrix(path_embeddings_matrix="okcupid_profiles_preprocessed.npy"):
    """
    Load the preprocessed embeddings matrix.
    """
    embeddings_matrix = np.load(path_embeddings_matrix)
    return embeddings_matrix


@st.cache_data(show_spinner=False)
def load_model(model="all-MiniLM-L6-v2"):
    """
    Load the SBERT model.
    """
    sbert_model = SentenceTransformer(model)
    return sbert_model


@st.cache_data(show_spinner=False)
def generate_top_matches_result(input_dict,
                                top_n,
                                path_embeddings_matrix="okcupid_profiles_preprocessed.npy",
                                model="all-MiniLM-L6-v2",
                                path_user_data="okcupid_profiles_cleaned.csv",
                                **kwarg):
    """
    Generate the top matches for the user input.
    """
    bar = st.progress(0, text="Operation is started.")

    # Check for invalid input values
    if invalid_input := check_invalid_input(input_dict):
        return invalid_input
    bar.progress(10, text='Step 1 of 9: Input data validated!')

    # Load the preprocessed embeddings matrix
    embeddings_matrix = load_embeddings_matrix(path_embeddings_matrix)
    # embeddings_matrix = np.load(path_embeddings_matrix)
    bar.progress(30, text='Step 2 of 9: Embeddings matrix loaded!')

    # Load the SBERT model
    sbert_model = load_model(model)
    # sbert_model = SentenceTransformer(model)
    bar.progress(40, text='Step 3 of 9: SBERT model loaded!')

    # Generate embeddings for the user input
    embeddings = input_embedding(input_dict, sbert_model, path_user_data)
    bar.progress(50, text='Step 4 of 9: Embeddings generated!')

    # Calculate the similarity between the user input and all profiles in the dataset
    similarity = similarity_input(embeddings, embeddings_matrix)
    bar.progress(60, text='Step 5 of 9: Similarity calculated!')

    # Convert the similarity array to a DataFrame
    similarity_df = pd.DataFrame(similarity)
    bar.progress(70, text='Step 6 of 9: Similarity DataFrame created!')

    # Load the user data
    user_df = pd.read_csv(path_user_data)
    bar.progress(80, text='Step 7 of 9: User data loaded!')

    # Generate the top matches for the user input
    result = get_top_matches(pd.DataFrame(
        [input_dict]), similarity_df, user_df, top_n, **kwarg)
    bar.progress(90, text='Step 8 of 9: Top matches generated!')

    # set the index to start from 1
    result = result.set_index(np.arange(1, len(result) + 1), drop=True)
    result.index.name = "Rank"
    bar.progress(100, text='Step 9 of 9: Operation completed!')

    return result

# Sample Testing Data
# input_dict = {
#     'age': 22.0,
#     'status': 'single',
#     'sex': 'female',
#     'orientation': 'gay',
#     'body_type': 'a little extra',
#     'diet': 'strictly anything',
#     'drinks': 'socially',
#     'drugs': 'never',
#     'education': 'working on college/university',
#     'ethnicity': 'asian, white',
#     'height': 190.0,
#     'job': 'transportation',
#     'location': 'south san francisco, california',
#     'offspring': "doesn't have kids, but might want them",
#     'pets': 'likes dogs and likes cats',
#     'religion': 'agnosticism and very serious about it',
#     'sign': 'gemini',
#     'smokes': 'sometimes',
#     'speaks': 'english',
#     'essay_all': """about me:  i would love to think that i was some some kind of intellectual: either the dumbest smart guy, or the smartest dumb guy. can't say i
# can tell the difference. i love to talk about ideas and concepts. i forge odd metaphors instead of reciting cliches. like the simularities between a friend of mine's house and an underwater salt mine. my favorite word is salt by the way (weird choice i know). to me most things in life are
# better as metaphors. i seek to make myself a little better everyday, in some productively lazy way. got tired of tying my shoes. considered hiring a five year old, but would probably have to tie both of our shoes... decided to only wear leather shoes dress shoes.  about you:  you love to have really serious, really deep conversations about really silly stuff. you have to be willing to snap me out of a light hearted rant with a kiss. you don't have to be funny, but you have to be able to make me laugh. you should be able to bend spoons with your mind, and telepathically make me smile while i am still at work. you should love life, and be cool with just letting the wind blow. extra points for reading all this and guessing my favorite video game (no hints given yet). and lastly you have a good attention span.,currently working as an international agent for a freight forwarding company. import, export, domestic you know the works. online classes and trying to better myself in my free time. perhaps a hours worth of a good book or a video game on a lazy sunday.,making people laugh. ranting about a good salting. finding simplicity in complexity, and complexity in simplicity.,the way i look. i am a six foot half asian, half caucasian mutt. it makes it tough not to notice me, and for me to blend in.,books: absurdistan, the republic, of mice and men (only book that made me want to cry), catcher in the rye, the prince.  movies: gladiator, operation valkyrie, the producers, down periscope.  shows: the borgia, arrested development, game of thrones, monty python  music: aesop rock,
# hail mary mallon, george thorogood and the delaware destroyers, felt  food: i'm down for anything.,food. water. cell phone. shelter.,duality and
# humorous things,trying to find someone to hang out with. i am down for anything except a club.,i am new to california and looking for someone to
# wisper my secrets to.,you want to be swept off your feet! you are tired of the norm. you want to catch a coffee or a bite. or if you want to talk philosophy."""
# }

# result = generate_top_matches_result(input_dict, top_n=10,
#             path_embeddings_matrix='okcupid_profiles_preprocessed.npy',
#             path_user_data="okcupid_profiles_cleaned.csv",
#             age_range=[16, 120],
#             height_range=[50, 250])
