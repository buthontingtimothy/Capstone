# Description: This is the main file for the Streamlit app. It contains the user interface and the logic to call the recommendation function.

# Import necessary libraries
# data manipulation
import pandas as pd
# streamlit
import streamlit as st
# function to generate top matches from input.py
from input import generate_top_matches_result
# image
from PIL import Image
# torch
import torch
# os
import os


# Streamlit app
def main():
    # prevent __path__ error
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
    
    # Set image and the title
    image = Image.open("pic.png")
    st.image(image)
    st.title('Find Your Perfect Match on Earth')

    # Create two columns: left for user input, right for filters
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Personal info ")

        #age
        age_input = st.number_input("Enter your age:",min_value=16,max_value=120,step=1,value=None,placeholder="Type a number...")
        st.write(age_input)

        #height(cm)
        height_input = st.number_input("Enter your height (in cm):",min_value=50,max_value=250,step=1,value=None,placeholder="Type an integer...")
        st.write(height_input)

        #status
        choic_status = ('<select>', 'single', 'married')
        status_option = st.selectbox('Status', choic_status)
        if status_option == '<select>':    
            status_option = 'N/A'

        #sex
        choice_sex = ('<select>', 'Male', 'Female')
        sex_option = st.selectbox('Sex', choice_sex)
        if sex_option == '<select>':    
            sex_option = 'N/A'

        #orientation
        choice_orientation = ('<select>', 'straight', 'gay', 'bisexual')
        orientation_option = st.selectbox('Orientation', choice_orientation)
        if orientation_option == '<select>':    
            orientation_option = 'N/A'

        #body_type
        choice_body_type = ('<select>', 'skinny', 'thin', 'fit', 'average', 'athletic', 'jacked', 'curvy', 'a little extra', 'full figured', 'overweight', 'used up', 'rather not say')
        body_type_option = st.selectbox('Body type', choice_body_type)
        if body_type_option == '<select>':    
            body_type_option = 'N/A'

        #diet
        choice_diet = ('<select>', 'anything', 'vegetarian', 'vegan', 'halal', 'kosher', 'other')
        diet_option = st.selectbox('Diet', choice_diet)
        if diet_option == '<select>':    
            diet_option = 'N/A'

        #drinks
        choice_drinks = ('not at all', 'rarely', 'socially', 'often', 'very often', 'desperately')
        drinks_option = st.select_slider('Drinking habit', choice_drinks)

        #drugs
        choice_drugs = ('never', 'sometimes', 'often')
        drugs_option = st.select_slider('Do you use drugs? ', choice_drugs)

        #education
        choice_education = ('<select>', 'college', 'high_school', 'law_school', 'master', 'med_school', 'PhD', 'space_camp')
        education_option = st.selectbox('Education', choice_education)
        if education_option == '<select>':    
            education_option = 'N/A'

        #ethnicity(multi)
        choice_ethnicity = ['asian', 'white', 'black', 'indian', 'hispanic / latin', 'other', 'native american', 'pacific islander', 'middle eastern']
        ethnicity_option = ", ".join(st.multiselect('Ethnicity (you can select more than one):', choice_ethnicity))


        #job / industry
        choice_job = ['<select>', 'transportation', 'hospitality / travel', 'artistic / musical / writer', 'computer / hardware / software', 'banking / financial / real estate', 'entertainment / media', 'sales / marketing / biz dev', 'medicine / health', 'science / tech / engineering', 'executive / management', 'education / academia', 'clerical / administrative', 'construction / craftsmanship', 'political / government', 'law / legal services', 'military', 'student', 'other', 'unemployed', 'retired', 'rather not say']
        job_option = st.selectbox('Job / Industry', choice_job)
        if job_option == '<select>':    
            job_option = 'N/A'

        #offspring_have***
        choice_offspring_have = ['<select>', "doesn't have kids", 'has a kid', 'has kids']
        have_kids_option = st.selectbox('Do you have kids?', choice_offspring_have)
        if have_kids_option == '<select>':    
            have_kids_option = 'N/A'

        #offspring_want***
        choice_offspring_want = ['<select>', "doesn't want", "doesn't want more", 'might want', 'might want more', 'wants', 'wants more']
        want_kids_option = st.selectbox('Do you want (more) kids?', choice_offspring_want)
        if want_kids_option == '<select>':    
            want_kids_option = 'N/A'

        #pets_have(multi)
        choice_pets_have = ['has dogs', 'has cats', 'no pets']
        have_pets_option = ", ".join(st.multiselect('Do you have any dogs or cats? (you can select more than one):', choice_pets_have, max_selections=2))

        if 'no pets' in have_pets_option and ('has dogs' in have_pets_option or 'has cats' in have_pets_option):
            have_pets_option = 'N/A'
            st.warning("You cannot select any other options when 'no pets' is chosen.")


        #pets_like(multi)
        choice_pets_like = ['likes dogs', 'likes cats', 'neutral to pets', 'dislikes dogs', 'dislikes cats']
        like_pets_option = ", ".join(st.multiselect('What is your preference regarding dogs and cats? (you can select more than one):', choice_pets_like, max_selections=2))
        
        if 'likes dogs' in like_pets_option and 'dislikes dogs' in like_pets_option:
            like_pets_option = 'N/A'
            st.warning("You cannot select 'likes dogs' and 'dislikes dogs' at the same time.")
        if 'likes cats' in like_pets_option and 'dislikes cats' in like_pets_option:
            like_pets_option = 'N/A'
            st.warning("You cannot select 'likes cats' and 'dislikes cats' at the same time.")
        if 'neutral to pets' in like_pets_option and any(option in like_pets_option for option in ['likes dogs', 'likes cats', 'dislikes dogs', 'dislikes cats']):
            like_pets_option = 'N/A'            
            st.warning("You cannot select 'neutral to pets' along with any other preference.")


        #religion
        choice_religion = ['<select>', 'Agnosticism', 'Atheism', 'Buddhism', 'Catholicism', 'Christianity','Hinduism', 'Islam', 'Judaism', 'Other', 'Irreligion']
        choice_religion_atti = ['very serious about it', 'somewhat serious about it', 'not too serious about it', 'laughing about it']
        religion_option = st.selectbox('What is your religion? ', choice_religion)
        if religion_option == '<select>':    
            religion_option = 'N/A'
        if religion_option not in ["Agnosticism", "Atheism", "Irreligion"]:
            # Attitude to religion
            atti_religion_option = st.select_slider('How seriously do you take your religion? (if any):', options = choice_religion_atti)
        else:
            atti_religion_option = 'N/A'

        #sign
        choice_sign = ['<select>', 'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        zodiac_option = st.selectbox('What is your zodiac sign?', choice_sign)
        if zodiac_option == '<select>':    
            zodiac_option = 'N/A'

        #smokes
        choice_smokes = ['<select>', 'no', 'trying to quit', 'when drinking', 'sometimes', 'yes']
        smoke_option = st.selectbox('Do you smoke?', choice_smokes)
        if smoke_option == '<select>':    
            smoke_option = 'N/A'

        #speaks_language(multi)
        choice_language = ['afrikaans', 'albanian', 'ancient', 'arabic', 'armenian', 'basque', 'belarusan', 'bengali', 'breton', 'bulgarian', 'catalan', 'cebuano', 'chechen', 'chinese', 'croatian', 'czech', 'danish', 'dutch', 'english', 'esperanto', 'estonian', 'farsi', 'finnish', 'french', 'frisian', 'georgian', 'german', 'greek', 'gujarati', 'hawaiian', 'hebrew', 'hindi', 'hungarian', 'icelandic', 'ilongo', 'indonesian', 'irish', 'italian', 'japanese', 'khmer', 'korean', 'latin', 'latvian', 'lisp', 'lithuanian', 'malay', 'maori', 'mongolian', 'norwegian', 'occitan', 'persian', 'polish', 'portuguese', 'romanian', 'rotuman',  'russian', 'sanskrit', 'sardinian', 'serbian', 'sign', 'slovak', 'slovenian', 'spanish', 'swahili', 'swedish', 'tagalog', 'tamil',  'thai', 'tibetan', 'turkish', 'ukrainian', 'urdu', 'vietnamese', 'welsh', 'yiddish', 'other']
        language_option = ", ".join(st.multiselect('What language(s) do you speak?', choice_language))

        #essay
        essay_input = st.text_input("Tell me a little more about yourself", value="", max_chars= 200)
        if essay_input:
            st.write(f"{essay_input}")
    
    # Filters
    with col2:
        st.subheader("Filters")
        # Filter by age, height, language, ethnicity, religion, sign, and keyword
        age_range = st.slider("Age Range:", min_value=18, max_value=100, value=(18, 35))
        height_range = st.slider("Height Range (cm):", min_value=100, max_value=250, value=(150, 200))
        # location = st.text_input("Preferred Location (city, state, or country):")
        speaks = st.multiselect("Preferred Languages:", choice_language)
        ethnicity = st.multiselect("Preferred Ethnicity:", choice_ethnicity)
        religion = st.multiselect("Preferred Religion:", choice_religion)
        sign = st.multiselect("Preferred Astrological Sign:", choice_sign)
        keyword_filter = st.text_input("Keyword Filter (e.g., hobbies, interests):")

    st.subheader('- End of Data Input -')

    st.divider()

    #Summary
    st.subheader("Summary of your info:")
    st.write(f"Your age: {age_input if age_input else 'N/A'}")
    st.write(f"Your height: {height_input if height_input else 'N/A'} cm")
    st.write('Your status: ', status_option)
    st.write('Your sex: ', sex_option)
    st.write('Your orientation: ', orientation_option)
    st.write('Your body type: ', body_type_option)
    st.write('Your diet: ', diet_option)
    st.write('Your drinking habit: ', drinks_option)
    st.write('Drug use: ', drugs_option)
    st.write('Your education level: ', education_option)
    st.write('Your ethnicity: ', ethnicity_option)
    st.write('Your job / industry: ', job_option)
    st.write('Do you have kids? ', have_kids_option)
    st.write('Do you want (more) kids? ', want_kids_option)
    st.write("Do you have any dogs or cats? ", have_pets_option)
    st.write('What is your preference regarding dogs and cats? ', like_pets_option)
    st.write('Your religion: ', religion_option)
    if religion_option not in ["Agnosticism", "Atheism", "Irreligion"]:
            st.write('How seriously do you take your religion? ', atti_religion_option)
    st.write('Your zodiac sign: ', zodiac_option)
    st.write('Do you smoke? ', smoke_option)
    st.write('The language(s) you speak : ', language_option)
    st.write('Tell me a little more about yourself: ', essay_input)


    # Create a dictionary to store the user input
    input_dict = {
        'age': age_input,
        'status': status_option,
        'sex': sex_option,
        'orientation': orientation_option,
        'body_type': body_type_option,
        'diet': diet_option,
        'drinks': drinks_option,
        'drugs': drugs_option,
        'education': education_option,
        'ethnicity': ethnicity_option,
        'height': height_input,
        'job': job_option,
        'location': pd.NA,
        'offspring': ', '.join(filter(lambda x: x != 'N/A', [have_kids_option, want_kids_option])),
        'pets': ', '.join(filter(lambda x: x != 'N/A', [have_pets_option, like_pets_option])),
        'religion': religion_option,
        'sign': zodiac_option,
        'smokes': smoke_option,
        'speaks': language_option,
        'essay_all': essay_input
        }
    
    # Create a dictionary to store the filters
    filter_dict = {
        'age_range': age_range,
        'height_range': height_range,
        # 'location': location,
        'speaks': speaks,
        'ethnicity': ethnicity,
        'religion': religion,
        'sign': sign,
        'keyword_filter': keyword_filter
        }


# Get recommendations when the user clicks the button
    if st.button("Get Recommendations"):

       # Call the recommendation function
        recommendations = generate_top_matches_result(
            input_dict,
            top_n=10,
            path_embeddings_matrix='okcupid_profiles_preprocessed.npy',
            path_user_data="okcupid_profiles_cleaned.csv",
            **filter_dict
        )
        # check if recommendations is not None and display the recommendations
        if recommendations is not None:
            st.dataframe(recommendations, width=1000)
        else:
            st.error("No recommendations found. Please try again with different inputs.")

# Run the app only if this script is run directly (not imported)
if __name__ == "__main__":
    main()