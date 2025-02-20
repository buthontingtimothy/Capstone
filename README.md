# OkCupid Profiles Analysis and Match Recommendation

This repository contains a comprehensive analysis of the OkCupid profiles dataset, along with a recommendation system to find top matches based on user input. The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles), includes 60,000 records with structured information such as age, sex, orientation, and text data from open-ended descriptions.

## Dataset Overview

The dataset contains the following columns:

- **Demographic Information**: `age`, `status`, `sex`, `orientation`, `body_type`, `diet`, `drinks`, `drugs`, `education`, `ethnicity`, `height`, `income`, `job`, `last_online`, `location`, `offspring`, `pets`, `religion`, `sign`, `smokes`, `speaks`
- **Text Data**: `essay0` to `essay9` (open-ended descriptions)

## Repository Structure

The repository is organized into several notebooks and scripts, each serving a specific purpose in the data processing and recommendation pipeline:

1. **Data Cleanse (`Data Cleanse.ipynb`)**:
   - **Input**: `okcupid_profiles.csv`
   - **Output**: `okcupid_profiles_cleaned.csv`
   - **Tasks**:
     - Fill null values with default values for most columns.
     - Use KNN to impute null values for `body_type`, `education`, and `ethnicity`.
     - Convert height to centimeters.
     - Remove URLs from essays and combine `essay0` to `essay9` into one long sentence.
     - Save the cleaned data as a CSV file.

2. **Embedding (`Embedding.ipynb`)**:
   - **Input**: `okcupid_profiles_cleaned.csv`
   - **Output**: `okcupid_profiles_preprocessed.npy`
   - **Tasks**:
     - Generate a long sentence from demographic data (excluding `age` and `height`).
     - Combine demographic and essay data into a single sentence.
     - Convert sentences into embeddings using `SentenceTransformer("all-MiniLM-L6-v2")`.
     - Standardize numeric data (`age` and `height`).
     - Concatenate embeddings and standardized numeric data.
     - Save the result as a `.npy` file.

3. **Similarity (`Similarity.ipynb`)**:
   - **Input**: `okcupid_profiles_preprocessed.npy`
   - **Output**: `similarity_matrix.npy`
   - **Tasks**:
     - Generate a similarity matrix using cosine similarity from `sklearn`.
     - Save the similarity matrix as a `.npy` file.

4. **Test Match (`test_match.ipynb`)**:
   - **Input**: `okcupid_profiles_cleaned.csv`, `similarity_matrix.npy`
   - **Output**: None
   - **Tasks**:
     - Test the similarity matrix by calling the `get_top_matches` function from `find_match.py`.
     - Verify if the recommendations are accurate and meaningful.

5. **Find Match (`find_match.py`)**:
   - **Functionality**:
     - Returns the top N matches with optional filters.
     - Filters by `sex` and `sexual orientation` by default.
     - Additional fields can be filtered based on user input.

6. **Input Processing (`input.py`)**:
   - **Functionality**:
     - Combines data cleansing, embedding, and similarity calculation.
     - Returns the top N (default is 10) matches for a given user input.

7. **Streamlit App (`app.py`)**:
   - **Functionality**:
     - Provides a user-friendly interface for input and filtering.
     - Displays a picture (`pic.png`) on the web page.
     - Left column: Fields for user input.
     - Right column: Fields for user filtering.
     - Saves all inputs and filters in a dictionary and passes them to `generate_top_matches_result` from `input.py`.
     - Displays recommendations after clicking the "Get Recommendations" button.

8. **Requirements (`requirements.txt`)**:
   - Lists all the libraries required to run the code.

9. **Image (`pic.png`)**:
   - An image displayed in the Streamlit web application.

## Usage

Follow these steps to set up and run the OkCupid Profiles Analysis and Match Recommendation system:

### 1. **Clone the Repository**
   First, clone the repository to your local machine:
   ```bash
   git clone https://github.com/buthontingtimothy/Capstone.git
   cd Capstone
   ```

### 2. **Install Dependencies**
   Install the required Python libraries using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

### 3. **Download the Dataset**
   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles) and place the `okcupid_profiles.csv` file in the root directory of the repository.

### 4. **Run the Data Processing Pipeline**
   Execute the following notebooks in order to preprocess the data and generate embeddings:

   1. **Data Cleaning**:
      - Run `Data Cleanse.ipynb` to clean the dataset and save the output as `okcupid_profiles_cleaned.csv`.
   2. **Embedding Generation**:
      - Run `Embedding.ipynb` to generate embeddings and save the output as `okcupid_profiles_preprocessed.npy`.
   3. **Similarity Matrix Calculation**:
      - Run `Similarity.ipynb` to generate the similarity matrix and save it as `similarity_matrix.npy`.

### 5. **Test the Recommendation System**
   - Run `test_match.ipynb` to test the recommendation system using the cleaned dataset and similarity matrix.

### 6. **Run the Streamlit App**
   Launch the Streamlit web application to interact with the recommendation system:
   ```bash
   streamlit run app.py
   ```
   - The app will open in your browser.
   - **Left Column**: Enter your profile information (e.g., age, sex, orientation, etc.).
   - **Right Column**: Apply filters (e.g., preferred age range, diet, etc.).
   - Click the **"Get Recommendations"** button to see the top matches.

### 7. **Customize Input and Filters**
   - Modify the input fields and filters in the Streamlit app to explore different recommendations.
   - The app will dynamically update the results based on your inputs and filters.
