import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import math
import base64
import hmac
from datetime import date
from concurrent.futures import ThreadPoolExecutor
import gcs_utils
import category_builder

st.set_page_config(page_title='Face Quiz', page_icon='❓')

# Specify your GCS bucket name
GS_BUCKET_NAME = st.secrets["GS_CREDENTIALS"]["GS_BUCKET_NAME"]

PEOPLE_LIST_BLOB = "DATA/People_list.csv"

# Apply custom CSS
with open("Style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

QUESTIONS_PER_ROUND = 20
QUESTION_DURATION = 12  # Duration in seconds
BONUS_BASE = 1100
BONUS_OFFSET = 100
COEFFICIENT = -2

# Scoring variables
BASE_SCORES = {1: 500, 2: 700, 3: 1000}
INCREMENT = 0.2
MAX_STREAK = 5
PENALTY = 3

REFRESH_RATE = 0.05  #In seconds
SCORE_INCREASE_GIMMICK_DURATION = 10  #In seconds
LAMBDA = 5

TOTAL_FRAMES = SCORE_INCREASE_GIMMICK_DURATION / REFRESH_RATE

def format_title(title, size=40, weight='bold', align='left'):
    return f"<div style='font-size: {size}px; font-weight: {weight}; text-align: {align};'>{title}</div>"

def standardize_string_column(column: pd.Series) -> pd.Series:
    """
    Standardize a string column by stripping whitespaces and converting to lowercase.

    Parameters:
    - column: The column to be standardized.

    Returns:
    - Standardized column.
    """
    return column.str.strip().str.lower()

difficulty_map = {
    'easy': 1,
    'medium': 2,
    'hard': 3
}

def calculate_exponential_time_bonus(elapsed_time):
    bonus = BONUS_BASE * np.exp(COEFFICIENT * elapsed_time / QUESTION_DURATION) - BONUS_OFFSET
    return max(0, bonus)

def compute_next_display_score(display_score, score, remaining_frames, total_frames, lambda_=5):
    """
    Calculate the next display score based on exponential increase to the actual score.

    Parameters:
    - display_score: Current displayed score.
    - score: Actual score.
    - remaining_frames: Number of frames left.
    - total_frames: Total number of frames.
    - lambda_: Exponential rate.

    Returns:
    - next_display_score: The next score to display.
    """
    # Calculate the normalized time (i.e., fraction of elapsed frames relative to total frames)
    elapsed_frames = total_frames - remaining_frames
    normalized_time = elapsed_frames / total_frames
    
    # Calculate the desired score at this normalized time based on the exponential formula
    desired_score_for_normalized_time = display_score + (score - display_score) * (1 - math.exp(-lambda_ * normalized_time))
    
    return int(desired_score_for_normalized_time)

# Function to update multiplier and correct answers count
def update_multiplier(is_correct):
    if is_correct:
        # Increment streak and multiplier
        st.session_state.streak = min(st.session_state.streak + 1, MAX_STREAK)
        
        # Increment correct answers count
        st.session_state.correct_answers += 1
    else:
        # Apply penalty and reset streak
        st.session_state.streak = max(st.session_state.streak - PENALTY, 0)
    
    st.session_state.multiplier = 1 + (st.session_state.streak * INCREMENT)

# Function to initialize the app by loading the list of people
def init():
    load_table_from_gcs()

# Function to load the CSV file containing the list of people
def load_table_from_gcs():
    data = gcs_utils.download_from_gcs(GS_BUCKET_NAME, PEOPLE_LIST_BLOB)
    st.session_state.people_list = standardize_people_list(pd.read_csv(data))

def standardize_people_list(people_list):
    people_list['Gender'] = standardize_string_column(people_list['Gender'])
    people_list['Category'] = standardize_string_column(people_list['Category'])

    # Provenance columns; rows from before this feature default to 'original'
    if 'Source' not in people_list.columns:
        people_list['Source'] = 'original'
    if 'Created_At' not in people_list.columns:
        people_list['Created_At'] = ''
    people_list['Source'] = people_list['Source'].fillna('original')
    people_list['Created_At'] = people_list['Created_At'].fillna('')
    return people_list

def save_people_list(people_list):
    csv_bytes = people_list.to_csv(index=False).encode('utf-8')
    gcs_utils.upload_bytes_to_gcs(GS_BUCKET_NAME, PEOPLE_LIST_BLOB, csv_bytes, content_type="text/csv")

def load_image_base64_from_gcs(image_blob_name):
    try:
        image_bytes = gcs_utils.download_from_gcs(GS_BUCKET_NAME, image_blob_name)
        return base64.b64encode(image_bytes.read()).decode('UTF-8')
    except Exception as e:
        st.error(f"Could not load image from GCS at {image_blob_name}: {e}")
        return None


def load_images_in_parallel(paths, max_threads=20):
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        return list(executor.map(load_image_base64_from_gcs, paths))


def display_image_base64(image_base64, width_percent=100):
    """Wrap the base64 encoded image in HTML and return the string for markdown.
    
    Args:
    - image_base64 (str): The base64 encoded image string.
    - width_percent (int): The width percentage for the image. Default is 100.
    
    Returns:
    - str: The HTML string to use with markdown.
    """
    if image_base64:
        html = f"""
        <div class='custom-image-container'>
            <img class='custom-image' style='width:{width_percent}%' src='data:image/png;base64,{image_base64}'>
        </div>
        """
        return html
    else:
        st.warning("No image provided.")
        return ""

def new_game():
    st.session_state.state = 1
    st.session_state.question_start_times = {}  # Reset the start times

def probabilistic_round_selection(possible_questions, asked_df, categories_share, difficulty):
    difficulty_patterns = {1: [0.6, 0.3, 0.1], 2: [0.4, 0.3, 0.3], 3: [0.2, 0.3, 0.5]}
    difficulty_distribution = difficulty_patterns[difficulty]
    selected_df = pd.DataFrame(columns=possible_questions.columns)
    
    current_round_questions = []

    for _ in range(QUESTIONS_PER_ROUND):
        if not asked_df.empty:
            # If the possible_questions pool lacks a specific combination of category and difficulty level
            unique_combos_in_asked = asked_df[asked_df['Category'].isin(categories_share.keys())].groupby(['Category', 'Fame_Level']).size().reset_index()
            
            for _, row in unique_combos_in_asked.iterrows():
                if (row['Category'] not in possible_questions['Category'].unique()) or \
                   (row['Fame_Level'] not in possible_questions[possible_questions['Category'] == row['Category']]['Fame_Level'].unique()):
                    combo_to_revert = asked_df[(asked_df['Category'] == row['Category']) & 
                                               (asked_df['Fame_Level'] == row['Fame_Level']) &
                                               (~asked_df['Name'].isin(current_round_questions))]
                    possible_questions = pd.concat([possible_questions, combo_to_revert], ignore_index=True)
                    asked_df = asked_df.drop(combo_to_revert.index)

        if possible_questions.empty:
            break

        # Compute probabilities for each question
        possible_questions['Category_Prob'] = possible_questions['Category'].map(categories_share)
        possible_questions['Level_Prob'] = possible_questions['Fame_Level'].map({1: difficulty_distribution[0], 2: difficulty_distribution[1], 3: difficulty_distribution[2]})
        possible_questions['Same_Cat_Level_Count'] = possible_questions.groupby(['Category', 'Fame_Level'])['Name'].transform('count')
        possible_questions['Prob'] = possible_questions['Category_Prob'] * possible_questions['Level_Prob'] / possible_questions['Same_Cat_Level_Count']

        # Normalize the probabilities
        prob_sum = possible_questions['Prob'].sum()
        possible_questions['Prob'] = possible_questions['Prob'] / prob_sum

        # Pick a question based on the probabilities
        chosen_row = np.random.choice(possible_questions.index, p=possible_questions['Prob'].values)
        chosen_question = possible_questions.loc[chosen_row]

        # Add chosen question to selected_df and remove it from possible_questions
        selected_df = pd.concat([selected_df, chosen_question.to_frame().T], ignore_index=True)
        current_round_questions.append(chosen_question['Name'])
        
        # Also add it to asked_df
        asked_df = pd.concat([asked_df, chosen_question.to_frame().T], ignore_index=True)
        
        possible_questions = possible_questions.drop(chosen_row).reset_index(drop=True)
    
    return asked_df, selected_df


def generate_dummy_answers(selected_df, people_list, difficulty):
    # Define the difficulty patterns
    difficulty_patterns = {
        1: [0.6, 0.3, 0.1],
        2: [0.4, 0.3, 0.3],
        3: [0.2, 0.3, 0.5]
    }
    difficulty_distribution = difficulty_patterns[difficulty]
    
    # Dictionary to store dummy answers for each question
    dummy_answers = {}
    
    # Remove the questions of the current round from people_list
    people_list = people_list[~people_list['Name'].isin(selected_df['Name'])]
    
    # For optimization, prepare a DataFrame with unique categories and their counts
    unique_categories = people_list['Category'].unique()
    unique_categories_count = len(unique_categories)
    
    for _, question in selected_df.iterrows():
        # Filter candidates with the same gender as the current question (case-insensitive)
        candidates = people_list[people_list['Gender'] == question['Gender'].lower()]

        # Fall back to the whole list if there aren't enough same-gender candidates
        if len(candidates) < 3:
            candidates = people_list

        # Calculate category probabilities
        if unique_categories_count > 1:
            category_probs = np.where(candidates['Category'] == question['Category'].lower(), 0.7, 0.3 / (unique_categories_count - 1))
        else:
            category_probs = np.ones(len(candidates))

        # Calculate fame level probabilities
        level_probs = np.array([difficulty_distribution[level-1] for level in candidates['Fame_Level']])

        # Calculate the total probabilities
        probabilities = category_probs * level_probs

        # Normalize the probabilities
        probabilities /= probabilities.sum()

        # Randomly select three dummy answers
        dummy_names = np.random.choice(candidates['Name'], size=3, p=probabilities, replace=False)

        dummy_answers[question['Name']] = list(dummy_names)

    return dummy_answers

def start_quiz():
    # Initialize score and streak
    st.session_state.score = 0
    st.session_state.display_score = 0
    st.session_state.streak = 0
    st.session_state.multiplier = 1

    checkbox_dict = {category: st.session_state[f"cat_{category}"] for category in st.session_state.people_list['Category'].unique().tolist()}
        
    # Retrieve the selected difficulty from the session state
    selected_difficulty = difficulty_map[st.session_state.selected_difficulty]
    selected_categories = [key for key, value in checkbox_dict.items() if value]
    
    # Bug prevention right here!
    if len(selected_categories) == 0:
        with st.spinner("What did you expect?"):
            time.sleep(2)
        with st.spinner("Select at least one category to start the quiz!"):
            time.sleep(3)
        st.session_state.state = 1
        return

    # Filter all questions to only the selected categories
    st.session_state.all_questions = st.session_state.people_list[st.session_state.people_list['Category'].isin(selected_categories)]
    
    # Create possible_questions by removing previously asked questions from all_questions
    st.session_state.possible_questions = st.session_state.all_questions[~st.session_state.all_questions['Name'].isin(st.session_state.asked_df['Name'])]

    # Compute categories_share based on the all_questions (already filtered for selected categories)
    category_counts = st.session_state.all_questions['Category'].value_counts()
    total_counts = category_counts.sum()
    categories_share = category_counts / total_counts
    categories_share = categories_share.to_dict()

    # Run the corrected probabilistic_round_selection function
    st.session_state.asked_df, st.session_state.selected_df = probabilistic_round_selection(st.session_state.possible_questions.copy(),
                                                                                              st.session_state.asked_df,
                                                                                              categories_share,
                                                                                              selected_difficulty)

    # The pool can be smaller than a full round (e.g. a freshly created small category)
    st.session_state.num_questions = len(st.session_state.selected_df)
    if st.session_state.num_questions == 0:
        with st.spinner("No questions available for this selection!"):
            time.sleep(3)
        st.session_state.state = 1
        return

    st.session_state.dummy_answers = generate_dummy_answers(st.session_state.selected_df, st.session_state.people_list, selected_difficulty)

    # Shuffle the answer options once per question so reruns don't reorder the buttons
    st.session_state.question_options = {}
    for i in range(st.session_state.num_questions):
        question = st.session_state.selected_df.iloc[i]
        options = [question['Name']] + st.session_state.dummy_answers[question['Name']]
        random.shuffle(options)
        st.session_state.question_options[i] = options

    st.session_state.selected_df['Image_GCS_Path'] = 'DATA/Pictures/' + st.session_state.selected_df['Name'].str.replace(' ', '_') + '.png'
    image_paths = st.session_state.selected_df['Image_GCS_Path'].tolist()
    st.session_state.selected_df['Image_Base64'] = load_images_in_parallel(image_paths)

    st.session_state.answers = {}
    st.session_state.state = 2
    st.session_state.counter = 0
    st.session_state.correct_answers = 0

# Function to display the form for starting the quiz
def display_form():

    st.title("Face Quiz", anchor=False)

    st.write("\n")

    play_tab, create_tab, manage_tab = st.tabs(["Play", "➕ Create a category", "🛠️ Manage"])

    with play_tab:
        unique_categories = st.session_state.people_list['Category'].unique().tolist()

        with st.form(key='my_form'):
            # Display checkboxes for category selection
            # Prefix widget keys so user-created category names can't collide with app session keys
            checkbox_dict = {category: st.checkbox(category, key=f"cat_{category}", value=True) for category in unique_categories}

            # Display dropdown for difficulty selection with display values
            difficulty_levels = list(difficulty_map.keys())
            selected_difficulty = st.selectbox('Select Difficulty Level:', difficulty_levels, key="selected_difficulty", index=1)

            # Start Quiz button
            submit_button = st.form_submit_button(label='Start Quiz',
                                      on_click=start_quiz)

    with create_tab:
        display_create_category()

    with manage_tab:
        display_manage_categories()

# Function to display the "create your own category" tab
def display_create_category():

    # Show the outcome of a category creation completed on the previous run
    result = st.session_state.pop("new_category_result", None)
    if result:
        st.success(f"Added {result['added']} people to the new '{result['category']}' category — "
                   f"select it in the Play tab to try it out! 🎉")
        if result["skipped"]:
            with st.expander(f"{len(result['skipped'])} people were skipped"):
                for line in result["skipped"]:
                    st.write(f"- {line}")

    st.write("Type any topic (e.g. *tennis players*, *French actors*, *astronauts*) and the app will "
             "build a new quiz category for it: an AI model picks the people and writes the hints, "
             "their portraits are fetched from Wikipedia, and the faces are detected and cropped automatically.")

    category = st.text_input("Category", key="new_category_name", placeholder="e.g. tennis players")
    n_people = st.slider("Number of people to generate", min_value=9, max_value=30, value=18, step=3,
                         key="new_category_size")

    existing_categories = set(st.session_state.people_list['Category'].unique())
    category_clean = category.strip().lower()

    if st.button("Generate category", type="primary", disabled=not category_clean):
        if category_clean in existing_categories:
            st.warning(f"The category '{category_clean}' already exists. Pick another name, or play it from the Play tab.")
        else:
            build_new_category(category_clean, n_people)

# Function to display the admin tab for reviewing and removing categories
def display_manage_categories():

    admin_password = st.secrets.get("ADMIN_PASSWORD", "")
    if not admin_password:
        st.info("Category management is disabled. Set an ADMIN_PASSWORD secret to enable it.")
        return

    if not st.session_state.get("admin_unlocked"):
        password = st.text_input("Admin password", type="password", key="admin_password_input")
        if st.button("Unlock", key="admin_unlock"):
            if hmac.compare_digest(password, admin_password):
                st.session_state.admin_unlocked = True
                st.rerun()
            else:
                st.error("Wrong password.")
        return

    # Show the outcome of a deletion completed on the previous run
    result = st.session_state.pop("category_deleted_result", None)
    if result:
        st.success(f"Deleted the '{result['category']}' category ({result['people']} people, "
                   f"{result['pictures']} pictures removed from storage).")

    people_list = st.session_state.people_list

    st.subheader("Categories", anchor=False)
    overview = (people_list.groupby('Category')
                .agg(People=('Name', 'count'),
                     Source=('Source', lambda s: ' + '.join(sorted(set(s)))),
                     Created=('Created_At', 'max'))
                .reset_index())
    st.dataframe(overview, width='stretch', hide_index=True)

    user_categories = sorted(people_list.loc[people_list['Source'] == 'user', 'Category'].unique())
    if user_categories:
        st.write(f"User-created categories: **{', '.join(user_categories)}**")
    else:
        st.write("No user-created categories yet.")

    st.subheader("Delete a category", anchor=False)
    all_categories = sorted(people_list['Category'].unique())
    target = st.selectbox("Category to delete", all_categories, key="delete_category_select")
    st.caption("Removes the category's people from the quiz. Pictures are deleted from storage "
               "only for user-created entries; original pictures are kept.")
    confirmed = st.checkbox(f"Yes, permanently delete '{target}'", key="delete_category_confirm")
    if st.button("Delete category", type="primary", disabled=not confirmed, key="delete_category_button"):
        delete_category(target)

# Function to delete a category's rows and its user-created pictures
def delete_category(category):
    people_list = st.session_state.people_list
    doomed = people_list[people_list['Category'] == category]
    remaining = people_list[people_list['Category'] != category].reset_index(drop=True)

    try:
        save_people_list(remaining)
    except Exception as e:
        st.error(f"Could not save the updated people list: {e}")
        return

    # Only clean up pictures of user-created people who are gone from the list entirely
    deleted_pictures = 0
    for _, row in doomed.iterrows():
        if row['Source'] == 'user' and row['Name'] not in remaining['Name'].values:
            try:
                gcs_utils.delete_from_gcs(GS_BUCKET_NAME, category_builder.blob_name_for(row['Name']))
                deleted_pictures += 1
            except Exception:
                pass  # An orphaned picture is harmless; the list is already saved

    st.session_state.people_list = remaining
    # Drop removed people from the asked-questions history as well
    st.session_state.asked_df = st.session_state.asked_df[
        ~st.session_state.asked_df['Name'].isin(doomed['Name'])]

    st.session_state.category_deleted_result = {
        "category": category,
        "people": len(doomed),
        "pictures": deleted_pictures,
    }
    st.rerun()

# Function to run the full category creation pipeline and persist the results
def build_new_category(category, n_people):
    people_list = st.session_state.people_list

    with st.status(f"Building the '{category}' category...", expanded=True) as status:
        st.write("🤖 Asking the AI model for a list of people...")
        try:
            people = category_builder.generate_people_with_llm(category, people_list['Name'].tolist(), n=n_people)
        except Exception as e:
            status.update(label="Category generation failed", state="error")
            st.error(f"Could not generate the list of people: {e}")
            return

        if not people:
            status.update(label="Category generation failed", state="error")
            st.error("The AI model did not return any usable people for this category. Try rephrasing it.")
            return

        st.write(f"🖼️ Fetching and cropping pictures for {len(people)} people from Wikipedia...")
        progress_bar = st.progress(0.0)
        added, skipped = [], []
        for i, person in enumerate(people):
            png_bytes, failure_reason = category_builder.build_person_picture(person['name'])
            if png_bytes is None:
                skipped.append(f"{person['name']}: {failure_reason}")
            else:
                try:
                    gcs_utils.upload_bytes_to_gcs(GS_BUCKET_NAME,
                                                  category_builder.blob_name_for(person['name']),
                                                  png_bytes,
                                                  content_type="image/png")
                    added.append(person)
                except Exception as e:
                    skipped.append(f"{person['name']}: picture upload failed ({e})")
            progress_bar.progress((i + 1) / len(people), text=person['name'])

        if len(added) < 4:
            status.update(label="Category generation failed", state="error")
            st.error("Not enough usable pictures were found to build this category. Try a different topic.")
            return

        st.write("💾 Saving the new category...")
        new_rows = pd.DataFrame([{
            'Name': person['name'],
            'Gender': person['gender'],
            'Category': category,
            'Fame_Level': person['fame_level'],
            'Short_Description': person['short_description'],
            'Source': 'user',
            'Created_At': date.today().isoformat(),
        } for person in added])

        try:
            updated_list = standardize_people_list(pd.concat([people_list, new_rows], ignore_index=True))
            save_people_list(updated_list)
        except Exception as e:
            status.update(label="Category generation failed", state="error")
            st.error(f"Could not save the updated people list: {e}")
            return

        st.session_state.people_list = updated_list
        st.session_state.new_category_result = {
            "category": category,
            "added": len(added),
            "skipped": skipped,
        }

    # Rerun so the Play tab (rendered before this pipeline ran) picks up the new category
    st.rerun()

# Function to handle submit answer button click
def submit_answer(answer):

    # Calculate elapsed time inside this function
    elapsed_time = time.time() - st.session_state.question_start_times[st.session_state.counter]

    # Determine if the answer is correct
    is_correct = answer == st.session_state.selected_df.iloc[st.session_state.counter]['Name']

    base_score = round(BASE_SCORES[st.session_state.selected_df.iloc[st.session_state.counter]['Fame_Level']]) if is_correct else 0

    time_bonus = round(calculate_exponential_time_bonus(elapsed_time)) if is_correct else 0
    total_score = round(st.session_state.multiplier * (base_score + time_bonus)) if is_correct else 0
    
    # Store the score breakdown along with the elapsed time
    st.session_state.score_breakdown[st.session_state.counter] = {
        "base_score": round(base_score),
        "time_bonus": round(time_bonus),
        "multiplier": round(st.session_state.multiplier, 2),
        "total_score": round(total_score),
        "elapsed_time": round(elapsed_time, 2)  # Rounded to 2 decimal places
    }

    update_multiplier(is_correct)

    # Update the total score with the rounded total_score calculated for this question
    st.session_state.score += round(total_score)

    # Store the answer
    st.session_state.answers[st.session_state.counter] = answer

    # Move to the next question or end the quiz if it was the last question
    if st.session_state.counter < st.session_state.num_questions - 1:
        st.session_state.counter += 1
    else:
        st.session_state.state = 3

# Function to display the current question
def display_question():

    st.empty()

    # Display the total score
    col1, col2, col3 = st.columns([7,3,3])

    col2.markdown(format_title("Score:",align='right'), unsafe_allow_html=True)

    score_placeholder = col3.empty()
    score_placeholder.markdown(format_title(st.session_state.display_score,align='right'), unsafe_allow_html=True)

    progress_bar = st.empty()
    progress_bar.progress(1.0) 
  
    # Get the current question
    question = st.session_state.selected_df.iloc[st.session_state.counter]
    
    # Display the question
    col1.markdown(format_title(f"{st.session_state.counter+1}/{st.session_state.num_questions}"),unsafe_allow_html=True)

    cols = st.columns([5,5])

    cols[0].markdown(display_image_base64(question['Image_Base64'],70), unsafe_allow_html=True)

    # Retrieve the pre-shuffled answers for this question
    options = st.session_state.question_options[st.session_state.counter]

    # Initialize question_start_times dictionary in the session state if not present
    if "question_start_times" not in st.session_state:
        st.session_state.question_start_times = {}

    # Check if a start time for the current question exists
    if st.session_state.counter not in st.session_state.question_start_times:
        st.session_state.question_start_times[st.session_state.counter] = time.time()

    cols[1].subheader("Who is this person?", anchor=False)
    cols[1].markdown("\n")
    colz = cols[1].columns(2)

    # Get the answer from the user
    for i in range(4):
        colz[(i//2)].button(options[i], key=f"option_{i+1}", on_click=submit_answer, args=(options[i],), width='stretch')

    st.markdown(f"{question['Short_Description']}")

    while True:
        # Calculate elapsed time using the start time for the current question
        elapsed_time = time.time() - st.session_state.question_start_times[st.session_state.counter]

        # Calculate and clamp the progress value to be between 0 and 1
        progress_value = max(0.0, min(1.0, 1.0 - (elapsed_time / QUESTION_DURATION)))

        # Update the progress bar
        progress_bar.progress(progress_value)

        # Calculate the number of remaining frames
        remaining_frames = (QUESTION_DURATION - elapsed_time) / REFRESH_RATE
        total_frames = QUESTION_DURATION / REFRESH_RATE

        # Calculate the next display score using the exponential function
        st.session_state.display_score = compute_next_display_score(st.session_state.display_score, st.session_state.score, remaining_frames, total_frames, lambda_=5)
        
        # Display the score
        score_placeholder.markdown(format_title(st.session_state.display_score,align='right'), unsafe_allow_html=True)

        # Check if QUESTION_DURATION seconds have passed
        if elapsed_time > QUESTION_DURATION:

            # Store the lack of answer
            submit_answer("No Answer")
            st.rerun()
        
        # Check if an answer is selected
        elif st.session_state.counter in st.session_state.answers:
            st.rerun()
        
        # Sleep before updating again
        time.sleep(REFRESH_RATE)


# Function to display the quiz results
def display_results():

    st.empty()

    num_questions = st.session_state.num_questions

    if st.session_state.correct_answers == num_questions:
        st.balloons()
        st.subheader(f"Congratulations! You answered **all** {num_questions} questions correctly.", anchor=False)
    else:
        # Display correct answer count and success rate
        st.write(f"You answered {st.session_state.correct_answers} out of {num_questions} questions correctly ({int(st.session_state.correct_answers/num_questions*100)}%).")

    # Display the total score
    col1, colx, col2, col3 = st.columns([3, 3, 5, 3])

    col1.markdown("\n")
    new_game_button = col1.button('New game', on_click=new_game, width='stretch')

    col2.markdown(format_title("Your score:", align='right'), unsafe_allow_html=True)

    score = st.session_state.score
    # Format the score with spaces as thousands separator
    formatted_score = "{:,}".format(score).replace(",", " ")

    # Display the formatted score using a title
    col3.markdown(format_title(formatted_score,align='right'), unsafe_allow_html=True)

    # Display each question, the user's answer, and the correct answer
    for i in range(num_questions):

        st.write("-----")

        col1, col2, col3 = st.columns([3,8,3])
        image = st.session_state.selected_df.iloc[i]['Image_Base64']
        name = st.session_state.selected_df.iloc[i]['Name']
        text = st.session_state.selected_df.iloc[i]['Short_Description']

        breakdown = st.session_state.score_breakdown[i]

        col1.markdown(display_image_base64(image,90), unsafe_allow_html=True)

        if breakdown['total_score'] != 0:
            col2.subheader(f":white_check_mark: {name}",anchor=False)
            helper = f"""(Question: **{breakdown['base_score']}** 
        + Speed bonus: **{breakdown['time_bonus']}**) 
        x Streak multiplier: **{breakdown['multiplier']}**"""
            col3.subheader(f"**{breakdown['total_score']}**",anchor=False, help=helper)
            col3.write(f"{breakdown['elapsed_time']} seconds")
        else:
            col2.subheader(f":x: {name}",anchor=False)
            helper = f"~~{st.session_state.answers[i]}~~"
            col3.subheader(f"**{breakdown['total_score']}**",anchor=False)
            col3.write(helper)

        col2.write(f"{text}")

# Function to handle reset button click
def reset():
    st.session_state.clear()

# Main function
def main():

    # Initialize session state variables
    if "people_list" not in st.session_state:
        try:
            init()
        except Exception as e:
            st.error(f"Could not load the people list from storage: {e}")
            st.stop()

    if "asked_df" not in st.session_state:
        st.session_state.asked_df = pd.DataFrame(columns=st.session_state.people_list.columns)

    if "state" not in st.session_state:
        st.session_state.state = 1

    if "score_breakdown" not in st.session_state:
        st.session_state.score_breakdown = {}

    if "answer" not in st.session_state:
        st.session_state.answer = ""
    
    # Initialize question_start_times dictionary in the session state if not present
    if "question_start_times" not in st.session_state:
        st.session_state.question_start_times = {}

    if st.session_state.state == 1:
        # State 1: Initial State
        display_form()

    elif st.session_state.state == 2:
        # State 2: Question State
        display_question()

    elif st.session_state.state == 3:
        # State 3: Finish State
        display_results()


# Call the main function
if __name__ == "__main__":
    main()