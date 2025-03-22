import dis
import pandas as pd
import streamlit as st
from PIL import Image
from input_backend import generate_top_matches_result

limit_recommendation_min, limit_recommendation_max = (10, 100)
limit_age_min, limit_age_max = (16, 120)
limit_height_min, limit_height_max = (50, 250)
choic_status = ('single', 'married')
choice_sex = ('male', 'female')
choice_orientation = ('straight', 'gay', 'bisexual')
choice_body_type = ('skinny', 'thin', 'fit', 'average', 'athletic', 'jacked', 'curvy',
                    'a little extra', 'full figured', 'overweight', 'used up', 'rather not say')
choice_diet = ('anything', 'vegetarian', 'vegan', 'halal', 'kosher', 'other')
choice_drinks = ('not at all', 'rarely', 'socially',
                 'often', 'very often', 'desperately')
choice_drugs = ('never', 'sometimes', 'often')
choice_education = ('college', 'high_school', 'law_school',
                    'master', 'med_school', 'PhD', 'space_camp')
choice_ethnicity = ('asian', 'white', 'black', 'indian', 'hispanic / latin', 'other', 'native american',
                    'pacific islander', 'middle eastern')
choice_job = ('transportation', 'hospitality / travel', 'artistic / musical / writer',
              'computer / hardware / software', 'banking / financial / real estate', 'entertainment / media',
              'sales / marketing / biz dev', 'medicine / health', 'science / tech / engineering',
              'executive / management', 'education / academia', 'clerical / administrative',
              'construction / craftsmanship', 'political / government', 'law / legal services',
              'military', 'student', 'other', 'unemployed', 'retired', 'rather not say')
choice_offspring_have = ("doesn't have kids", 'has a kid', 'has kids')
choice_offspring_want = ("doesn't want", "doesn't want more",
                         'might want', 'might want more', 'wants', 'wants more')
choice_pets_have = ('has dogs', 'has cats', 'no pets')
choice_pets_like = ('likes dogs', 'likes cats',
                    'neutral to pets', 'dislikes dogs', 'dislikes cats')
choice_religion = ('Agnosticism', 'Atheism', 'Buddhism', 'Catholicism', 'Christianity', 'Hinduism', 'Islam',
                   'Judaism', 'Other', 'Irreligion')
choice_religion_atti = ('very serious about it', 'somewhat serious about it',
                        'not too serious about it', 'laughing about it')
choice_sign = ('Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
               'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces')
choice_smokes = ('no', 'trying to quit', 'when drinking', 'sometimes', 'yes')
choice_language = ('afrikaans', 'albanian', 'ancient', 'arabic', 'armenian', 'basque', 'belarusan', 'bengali', 'breton',
                   'bulgarian', 'catalan', 'cebuano', 'chechen', 'chinese', 'croatian', 'czech', 'danish', 'dutch',
                   'english', 'esperanto', 'estonian', 'farsi', 'finnish', 'french', 'frisian', 'georgian', 'german',
                   'greek', 'gujarati', 'hawaiian', 'hebrew', 'hindi', 'hungarian', 'icelandic', 'ilongo', 'indonesian',
                   'irish', 'italian', 'japanese', 'khmer', 'korean', 'latin', 'latvian', 'lisp', 'lithuanian', 'malay',
                   'maori', 'mongolian', 'norwegian', 'occitan', 'persian', 'polish', 'portuguese', 'romanian', 'rotuman',
                   'russian', 'sanskrit', 'sardinian', 'serbian', 'sign', 'slovak', 'slovenian', 'spanish', 'swahili',
                   'swedish', 'tagalog', 'tamil',  'thai', 'tibetan', 'turkish', 'ukrainian', 'urdu', 'vietnamese',
                   'welsh', 'yiddish', 'other')


def header():
    image = Image.open("pic.png")
    st.image(image)
    st.title('Find Your Perfect Match on Earth')


def offsprings_input():
    # Get selections directly
    have_kids_option = st.selectbox(
        'Do you have kids?', choice_offspring_have, index=None, placeholder='<select>')
    want_kids_option = st.selectbox(
        'Do you want (more) kids?', choice_offspring_want, index=None, placeholder='<select>')

    # Combine results
    kids_info = []
    if have_kids_option:
        kids_info.append(have_kids_option)
    if want_kids_option:
        kids_info.append(want_kids_option)

    return " and ".join(kids_info) if kids_info else None


def pet_input():
    # Pet Ownership Section
    selected_have = st.multiselect(
        'Do you have any dogs or cats? (you can select more than one):',
        choice_pets_have,
        placeholder='<select>',
        max_selections=2
    )

    # Process pet ownership with local variable
    have_pets = None
    if selected_have:
        if 'no pets' in selected_have and len(selected_have) > 1:
            st.warning("Can't select 'no pets' with other options", icon="⚠️")
        else:
            have_pets = selected_have[0] if len(
                selected_have) == 1 else ", ".join(selected_have)

    # Pet Preference Section
    selected_like = st.multiselect(
        'What is your preference regarding dogs and cats? (you can select more than one):',
        choice_pets_like,
        placeholder='<select>',
        max_selections=2
    )

    # Process preferences with local variable
    like_pets = None
    if selected_like:
        conflicts = [
            {'likes dogs', 'dislikes dogs'}.issubset(selected_like),
            {'likes cats', 'dislikes cats'}.issubset(selected_like),
            'neutral to pets' in selected_like and len(selected_like) > 1
        ]

        if any(conflicts):
            st.warning("Invalid combination of preferences", icon="⚠️")
        else:
            like_pets = selected_like[0] if len(
                selected_like) == 1 else ", ".join(selected_like)

    # Combine results
    pets_info = []
    if have_pets:
        pets_info.append(have_pets)
    if like_pets:
        pets_info.append(like_pets)

    return " and ".join(pets_info) if pets_info else None


def reglion_input():
    # Use local variables instead of session state
    religion = st.selectbox('What is your religion? ',
                            choice_religion, index=None, placeholder='<select>')
    attitude = None

    # Handle religion seriousness slider
    if religion in ('Buddhism', 'Catholicism', 'Christianity', 'Hinduism', 'Islam', 'Judaism'):
        disabled = False
        attitude = st.select_slider('How seriously do you take your religion? (if any):',
                                    options=choice_religion_atti
                                    )

    # Handle "Other" religion case
    if religion == 'Other':
        attitude = st.text_input(
            "Please specify your religion", value=None, max_chars=50)

    # Combine results
    religion_info = []
    if religion:
        religion_info.append(religion)
    if attitude:
        religion_info.append(attitude)

    return " and ".join(religion_info) if religion_info else None


@st.fragment
def personal_input():
    st.subheader("Personal info ")
    st.write("Please fill in to find your perfect match:")

    # Initialize input_dict if it doesn't exist
    st.session_state.setdefault('input_dict', {})
    i_dict = st.session_state.input_dict

    # Store values directly in input_dict
    i_dict['age'] = st.number_input(
        "Enter your age:",
        min_value=limit_age_min,
        max_value=limit_age_max,
        step=1,
        value=None,
        placeholder="Type a number..."
    )

    i_dict['height'] = st.number_input(
        "Enter your height (in cm):",
        min_value=limit_height_min,
        max_value=limit_height_max,
        step=1,
        value=None,
        placeholder="Type a number..."
    )

    i_dict['status'] = st.selectbox(
        'Status',
        choic_status,
        index=None,
        placeholder='<select>'
    )

    i_dict['sex'] = st.selectbox(
        'Sex',
        choice_sex,
        index=None,
        placeholder='<select>'
    )

    i_dict['orientation'] = st.selectbox(
        'Orientation',
        choice_orientation,
        index=None,
        placeholder='<select>'
    )

    i_dict['body_type'] = st.selectbox(
        'Body type',
        choice_body_type,
        index=None,
        placeholder='<select>'
    )

    i_dict['diet'] = st.selectbox(
        'Diet',
        choice_diet,
        index=None,
        placeholder='<select>'
    )

    i_dict['drinks'] = st.select_slider(
        'Drinking habit',
        choice_drinks
    )

    i_dict['drugs'] = st.select_slider(
        'Do you use drugs? ',
        choice_drugs
    )

    i_dict['education'] = st.selectbox(
        'Education',
        choice_education,
        index=None,
        placeholder='<select>'
    )

    i_dict['ethnicity'] = None if not (
        temp_ethnicity := ", ".join(
            st.multiselect(
                'Ethnicity (you can select more than one):',
                choice_ethnicity,
                default=None,
                placeholder='<select>'
            )
        )
    ) else temp_ethnicity

    i_dict['job'] = st.selectbox(
        'Job / Industry',
        choice_job,
        index=None,
        placeholder='<select>'
    )

    i_dict['offspring'] = offsprings_input()

    i_dict['pets'] = pet_input()

    i_dict['religion'] = reglion_input()

    i_dict['sign'] = st.selectbox(
        'What is your zodiac sign?',
        choice_sign,
        index=None,
        placeholder='<select>'
    )

    i_dict['smokes'] = st.selectbox(
        'Do you smoke?',
        choice_smokes,
        index=None,
        placeholder='<select>'
    )

    i_dict['speaks'] = None if not (
        temp_speaks := ", ".join(
            st.multiselect(
                'What language(s) do you speak?',
                choice_language,
                default=None,
                placeholder='<select>'
            )
        )
    ) else temp_speaks

    i_dict['essay_all'] = st.text_input(
        "Tell me a little more about yourself",
        value=None,
        max_chars=200,
        placeholder="Type a keyword..."
    )

    # Set default value for location
    i_dict['location'] = pd.NA

    return st.session_state.input_dict


@st.fragment
def filter_input():

    # Initialize filter_dict in session state
    default_filter_dict = {
        'age_range': (limit_age_min, limit_age_max),
        'height_range': (limit_height_min, limit_height_max),
        'location': None,
        'speaks': None,
        'ethnicity': None,
        'religion': None,
        'sign': None,
        'keyword_filter': None
    }
    if 'filter_dict' not in st.session_state:
        st.session_state.filter_dict = default_filter_dict.copy()
    f_dict = st.session_state.filter_dict

    st.subheader("Filters (Optional)")
    with st.expander("Show Filters"):
        f_dict['age_range'] = st.slider(
            "Age Range:", min_value=limit_age_min, max_value=limit_age_max, value=f_dict['age_range'])
        f_dict['height_range'] = st.slider(
            "Height Range (cm):", min_value=limit_height_min, max_value=limit_height_max, value=f_dict['height_range'])
        f_dict['location'] = st.text_input(
            "Preferred Location (city, state, or country):", value=f_dict['location'], placeholder="Type a location...")
        f_dict['speaks'] = st.multiselect(
            "Preferred Languages:", options=choice_language, default=f_dict['speaks'])
        f_dict['ethnicity'] = st.multiselect(
            "Preferred Ethnicity:", options=choice_ethnicity, default=f_dict['ethnicity'])
        f_dict['religion'] = st.multiselect(
            "Preferred Religion:", options=choice_religion, default=f_dict['religion'])
        f_dict['sign'] = st.multiselect(
            "Preferred Astrological Sign:", options=choice_sign, default=f_dict['sign'])
        f_dict['keyword_filter'] = st.text_input(
            "Keyword Filter (e.g., hobbies, interests):", value=f_dict['keyword_filter'], placeholder="Type a keyword...")

        # Reset to defaults
        if st.button('Clear Filters'):
            st.session_state.filter_dict = default_filter_dict.copy()

    return st.session_state.filter_dict


def filter_dict_split(df):
    # Make a copy of the DataFrame
    df = df.copy()
    # Split columns ending with '_range' into '_min' and '_max'
    new_columns = []  # Will store split columns in order
    columns_to_drop = []

    for col in df.columns:
        if '_range' in col:
            # Generate new column names
            min_col = col.replace('_range', '_min')
            max_col = col.replace('_range', '_max')

            # Split tuple into separate columns
            df[[min_col, max_col]] = pd.DataFrame(
                df[col].tolist(), index=df.index)

            columns_to_drop.append(col)
            new_columns.extend([min_col, max_col])  # Track order

    # Remove original range columns
    df.drop(columns=columns_to_drop, inplace=True)

    # Get non-range columns (preserve their original order)
    original_non_range_columns = [
        col for col in df.columns if col not in new_columns]

    # Reorder DataFrame with new columns first
    df = df[new_columns + original_non_range_columns]
    return df


@st.fragment
def select_recommendation():
    st.session_state.top_n = st.slider(
        'Number of Recommendation', min_value=limit_recommendation_min, max_value=limit_recommendation_max, value=limit_recommendation_min, step=5)


def recommendation_result(result):
    st.session_state.recommendations = result
    # check if recommendations is not None and display the recommendations
    if isinstance(st.session_state.recommendations, pd.DataFrame):
        if len(st.session_state.recommendations) == st.session_state.top_n:
            st.success(
                f"Here are your top {st.session_state.top_n} recommendations:")
            st.dataframe(st.session_state.recommendations)
        elif len(st.session_state.recommendations) == 0:
            st.error(
                "No recommendations found. Please try again with different inputs.")
        else:
            st.warning(
                f"Only {len(st.session_state.recommendations)} recommendation(s) found. Please try again with different inputs.")
            st.dataframe(st.session_state.recommendations)

        # Display User input and Filters
        st.write("Your input:")
        st.write(pd.DataFrame([st.session_state.input_dict]))
        st.write("Your filters:")

        # Split columns ending with '_range' into '_min' and '_max'
        st.write(filter_dict_split(
            pd.DataFrame([st.session_state.filter_dict])))

        # Display a thank you message
        st.write("Thank you for using our service!")

    else:
        st.error(
            f"{st.session_state.recommendations}. No recommendations found. Please try again with different inputs.")


@st.fragment
def get_recommendation():
    # let the user select the number of recommendations without affecting the other inputs
    select_recommendation()
    # Get recommendations when the user clicks the button
    st.session_state.submit = st.button('Get Recommendations')
    if st.session_state.submit:
       # Call the recommendation function
        st.session_state.recommendations = generate_top_matches_result(
            input_dict=st.session_state.input_dict,
            top_n=st.session_state.top_n,
            **st.session_state.filter_dict
        )
        # Display the recommendation result
        recommendation_result(st.session_state.recommendations)
