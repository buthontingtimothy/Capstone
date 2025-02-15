import pandas as pd

def get_top_matches(user_id, similarity_df, user_df, top_n=5, 
                    age_range=None, height_range=None, location=None, 
                    education=None, job=None, pets=None, offspring=None, 
                    speaks=None, ethnicity=None, diet=None, 
                    body_type=None, drinks=None, drugs=None, 
                    religion=None, sign=None, smokes=None, 
                    keyword_filter=None):
    """
    Returns top N matches for a given user, with optional flexible filters.

    Parameters:
        user_id (int): ID of the user to find matches for.
        similarity_df (pd.DataFrame): Pairwise similarity matrix.
        user_df (pd.DataFrame): Original user data.
        top_n (int): Number of matches to return.
        Various filters: Accepts category names or partial text matching.

    Returns:
        pd.DataFrame: Top N matching users.
    """

    # Make a copy of the user_df to avoid modifying the original
    user_df = user_df.copy()

    # Get user's gender and orientation
    user_sex = user_df.loc[user_id, "sex"]
    user_orientation = user_df.loc[user_id, "orientation"]

    # Define valid matches based on gender + orientation
    if user_sex == "male":
        if user_orientation == "straight":
            valid_matches = user_df[(user_df["sex"] == "female") & 
                                    (user_df["orientation"].isin(["straight", "bisexual"]))]
        elif user_orientation == "gay":
            valid_matches = user_df[(user_df["sex"] == "male") & 
                                    (user_df["orientation"].isin(["gay", "bisexual"]))]
        elif user_orientation == "bisexual":
            valid_matches = user_df[((user_df["sex"] == "female") & 
                                     (user_df["orientation"].isin(["straight", "bisexual"]))) |
                                    ((user_df["sex"] == "male") & 
                                     (user_df["orientation"].isin(["gay", "bisexual"])))]
    elif user_sex == "female":
        if user_orientation == "straight":
            valid_matches = user_df[(user_df["sex"] == "male") & 
                                    (user_df["orientation"].isin(["straight", "bisexual"]))]
        elif user_orientation == "gay":
            valid_matches = user_df[(user_df["sex"] == "female") & 
                                    (user_df["orientation"].isin(["gay", "bisexual"]))]
        elif user_orientation == "bisexual":
            valid_matches = user_df[((user_df["sex"] == "male") & 
                                     (user_df["orientation"].isin(["straight", "bisexual"]))) |
                                    ((user_df["sex"] == "female") & 
                                     (user_df["orientation"].isin(["gay", "bisexual"])))]
    else:
        return pd.DataFrame()  # No valid matches

    # Apply Numeric Filters
    if age_range:
        valid_matches = valid_matches[(valid_matches["age"] >= age_range[0]) & 
                                      (valid_matches["age"] <= age_range[1])]

    if height_range:
        valid_matches = valid_matches[(valid_matches["height"] >= height_range[0]) & 
                                      (valid_matches["height"] <= height_range[1])]

    # Flexible Location Filter (city, state, country)
    if location:
        valid_matches = valid_matches[valid_matches["location"].str.contains(location, case=False, na=False)]

    # Flexible Text-Based Filters
    text_filters = {
        "education": education, "job": job, "speaks": speaks, "ethnicity": ethnicity, "sign": sign, "religion": religion
    }
    
    for column, value in text_filters.items():
        if value:
            valid_matches = valid_matches[valid_matches[column].str.contains('|'.join(value), case=False, na=False)]

    # *🔹 Fuzzy Diet Matching*
    if diet:
        diet_synonyms = {
            "vegan": ["vegan", "vegetarian", "mostly vegetarian"],
            "vegetarian": ["vegetarian", "mostly vegetarian"],
            "pescatarian": ["pescatarian", "mostly pescatarian"],
            "halal": ["halal", "mostly halal"],
            "kosher": ["kosher", "mostly kosher"]
        }
        
        valid_matches = valid_matches[valid_matches["diet"].apply(
            lambda x: any(diet.lower() in diet_synonyms.get(d, [d.lower()]) for d in str(x).split(", ")) 
            if pd.notna(x) else False
        )]

    # *🔹 Flexible Pet Preferences*
    if pets:
        for pet_pref in pets:
            if pet_pref == "likes dogs":
                valid_matches = valid_matches[
                    valid_matches["pets"].str.contains("has dogs|likes dogs", case=False, na=False)
                ]
            elif pet_pref == "dislikes dogs":
                valid_matches = valid_matches[
                    valid_matches["pets"].str.contains("dislikes dogs", case=False, na=False)
                ]
            elif pet_pref == "likes cats":
                valid_matches = valid_matches[
                    valid_matches["pets"].str.contains("has cats|likes cats", case=False, na=False)
                ]
            elif pet_pref == "dislikes cats":
                valid_matches = valid_matches[
                    valid_matches["pets"].str.contains("dislikes cats", case=False, na=False)
                ]

    # *🔹 Flexible Offspring (Kids) Preferences*
    if offspring:
        if "likes kids" in offspring:
            valid_matches = valid_matches[valid_matches["offspring"].str.contains("has kids|wants kids", case=False, na=False)]
        if "dislikes kids" in offspring:
            valid_matches = valid_matches[~valid_matches["offspring"].str.contains("has kids|wants kids", case=False, na=False)]
        if "neutral about kids" in offspring:
            valid_matches = valid_matches[valid_matches["offspring"].isna()]

    # Exact Match Filters (Categories)
    category_filters = {
        "body_type": body_type, "drinks": drinks, "drugs": drugs, "smokes": smokes
    }

    for column, value in category_filters.items():
        if value:
            valid_matches = valid_matches[valid_matches[column] == value]

    # Get valid user IDs
    valid_user_ids = valid_matches.index

    # Get similarity scores for valid matches
    filtered_similarities = similarity_df.loc[user_id, valid_user_ids]

    # Apply keyword-based filtering (if provided)
    if keyword_filter:
        keyword_filtered_users = valid_matches[
            valid_matches["essay_all"].str.contains(keyword_filter, case=False, na=False)
        ].index
        filtered_similarities = filtered_similarities.loc[keyword_filtered_users]

    # Return top N matches
    top_matches = filtered_similarities.sort_values(ascending=False).head(top_n)
    
    return user_df.loc[top_matches.index].assign(similarity_score=top_matches.values)