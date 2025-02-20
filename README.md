Got it! If users only want to try the **recommendation system** and skip the EDA part (which involves `Data_Grouping.ipynb` and Tableau), we can make that clear in the **Usage** section. Here's the revised `README.md` with a streamlined **Usage** section for users who only want to use the recommendation system:

---

```markdown
# OkCupid Profiles Analysis and Match Recommendation

This repository provides a comprehensive analysis of the [OkCupid Profiles dataset](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles) and implements a recommendation system to find top matches based on user input. The dataset contains 60,000 records with structured information (e.g., age, sex, orientation) and text data from open-ended user descriptions.

## Dataset Overview

The dataset includes the following columns:

- **Demographic Information**: `age`, `status`, `sex`, `orientation`, `body_type`, `diet`, `drinks`, `drugs`, `education`, `ethnicity`, `height`, `income`, `job`, `last_online`, `location`, `offspring`, `pets`, `religion`, `sign`, `smokes`, `speaks`
- **Text Data**: `essay0` to `essay9` (open-ended user descriptions)

## Repository Structure

The repository is organized into the following files and notebooks:

### Notebooks
1. **`Data Cleanse.ipynb`**:
   - **Input**: `okcupid_profiles.csv`
   - **Output**: `okcupid_profiles_cleaned.csv`
   - **Tasks**:
     - Fill null values with default values for most columns.
     - Use KNN to impute null values for `body_type`, `education`, and `ethnicity`.
     - Convert height from inches to centimeters.
     - Remove URLs from essays and combine `essay0` to `essay9` into a single sentence.
     - Save the cleaned data as a CSV file.

2. **`Embedding.ipynb`**:
   - **Input**: `okcupid_profiles_cleaned.csv`
   - **Output**: `okcupid_profiles_preprocessed.npy`
   - **Tasks**:
     - Generate a long sentence from demographic data (excluding `age` and `height`).
     - Combine demographic and essay data into a single sentence.
     - Convert sentences into embeddings using `SentenceTransformer("all-MiniLM-L6-v2")`.
     - Standardize numeric data (`age` and `height`).
     - Concatenate embeddings and standardized numeric data.
     - Save the result as a `.npy` file.

3. **`Similarity.ipynb`**:
   - **Input**: `okcupid_profiles_preprocessed.npy`
   - **Output**: `similarity_matrix.npy`
   - **Tasks**:
     - Generate a similarity matrix using cosine similarity from `sklearn`.
     - Save the similarity matrix as a `.npy` file.

4. **`test_match.ipynb`**:
   - **Input**: `okcupid_profiles_cleaned.csv`, `similarity_matrix.npy`
   - **Output**: None
   - **Tasks**:
     - Test the similarity matrix by calling the `get_top_matches` function from `find_match.py`.
     - Verify if the recommendations are accurate and meaningful.

### Scripts
5. **`find_match.py`**:
   - **Functionality**:
     - Returns the top N matches with optional filters.
     - Filters by `sex` and `sexual orientation` by default.
     - Additional fields can be filtered based on user input.

6. **`input.py`**:
   - **Functionality**:
     - Combines data cleansing, embedding, and similarity calculation.
     - Returns the top N (default is 10) matches for a given user input.

7. **`app.py`**:
   - **Functionality**:
     - Provides a user-friendly interface using Streamlit for input and filtering.
     - Displays an image (`pic.png`) on the web page.
     - Left column: Fields for user input.
     - Right column: Fields for user filtering.
     - Saves all inputs and filters in a dictionary and passes them to `generate_top_matches_result` from `input.py`.
     - Displays recommendations after clicking the "Get Recommendations" button.

### Additional Files
8. **`requirements.txt`**:
   - Lists all the libraries required to run the code.

9. **`pic.png`**:
   - An image displayed in the Streamlit web application.

---

## Usage

### For Recommendation System Only
If you only want to use the **recommendation system**, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/buthontingtimothy/Capstone.git
   cd okcupid-profiles-analysis
   ```

2. **Install Dependencies**:
   Install the required Python libraries using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Preprocessed Data**:
   - Download the preprocessed files (`okcupid_profiles_cleaned.csv`, `okcupid_profiles_preprocessed.npy`, and `similarity_matrix.npy`) from the repository (if available) or generate them by running the following notebooks:
     - `Data Cleanse.ipynb`
     - `Embedding.ipynb`
     - `Similarity.ipynb`

4. **Run the Streamlit App**:
   Launch the Streamlit web application to interact with the recommendation system:
   ```bash
   streamlit run app.py
   ```
   - The app will open in your browser.
   - **Left Column**: Enter your profile information (e.g., age, sex, orientation, etc.).
   - **Right Column**: Apply filters (e.g., preferred age range, diet, etc.).
   - Click the **"Get Recommendations"** button to see the top matches.

---

## Exploratory Data Analysis (EDA) with Tableau (Optional)

If you're interested in exploring the dataset further, you can use the `Data_Grouping.ipynb` notebook and the Tableau dashboard:

1. **`Data_Grouping.ipynb`**:
   - **Input**: `okcupid_profiles_cleaned.csv`
   - **Output**: `okcupid_profiles_output.csv`
   - **Tasks**:
     - Group and transform the cleaned dataset for EDA purposes.

2. **Tableau Dashboard**:
   - The cleaned and grouped dataset (`okcupid_profiles_output.csv`) was used to create an interactive Tableau dashboard for exploratory data analysis.
   - **Link to Tableau Dashboard**: [OkCupid EDA Dashboard](https://public.tableau.com/app/profile/hon.ting.but/viz/EDA_17400899191730/Story)

---
