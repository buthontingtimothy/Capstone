# Description: This is the main file for the Streamlit app. It contains the user interface and the logic to call the recommendation function.

# Import necessary libraries
# data manipulation
import pandas as pd
# streamlit
import streamlit as st
# recommendation function

from input_frontend import header, personal_input, filter_input, get_recommendation
# torch
import torch
# os
import os


# Streamlit app
def main():
    # prevent __path__ error
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
    
    # Set image and the title
    header()
    
    # Create two columns: left for user input, right for filters
    col1, col2 = st.columns(2)

    # User input
    with col1:
        st.session_state.input_dict = personal_input()

    # Filters
    with col2:
        st.session_state.filter_dict = filter_input()

    st.subheader('- End of Data Input -')
    st.divider()

    # Select and get recommendation
    # select_recommendation()
    get_recommendation()

# Run the app only if this script is run directly (not imported)
if __name__ == "__main__":
    main()