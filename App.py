import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import math
import base64

# Set up the page configuration
st.set_page_config(page_title='Face Quiz', page_icon='❓')

# Apply custom CSS (with error handling in case the file is missing)
try:
    with open("Style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Style.css file not found. Using default styling.")

# Global constants
QUESTION_DURATION = 12  # Duration in seconds
BONUS_BASE = 1100
BONUS_OFFSET = 100
COEFFICIENT = -2

# Scoring variables
BASE_SCORES = {1: 500, 2: 700, 3: 1000}
INCREMENT = 0.2
MAX_STREAK = 5
PENALTY = 3

REFRESH_RATE = 0.05  # In seconds
SCORE_INCREASE_GIMMICK_DURATION = 10  # In seconds
LAMBDA = 5

TOTAL_FRAMES = SCORE_INCREASE_GIMMICK_DURATION / REFRESH_RATE

def format_title(title, size=40, weight='bold', align='left'):
    return f"<div style='font-size: {size}px; font-weight: {weight}; text-align: {align};'>{title}</div>"

def standardize_string_column(column: pd.Series) -> pd.Series:
    """
    Standardize a string column by stripping whitespaces and converting to lowercase.
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
    Calculate the next display score based on an exponential increase toward the actual score.
    """
    elapsed_frames = total_frames - remaining_frames
    normalized_time = elapsed_frames / total_frames
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

def init():
    load_table()

def load_table():
    """
    Load the people list CSV and apply string standardization.
    """
    try:
        st.session_state.people_list = pd.read_csv("DATA/People_list.csv")
    except FileNotFoundError:
        st.error("DATA/People_list.csv file not found. Please check the path.")
        st.stop()
    st.session_state.people_list['Gender'] = standardize_string_column(st.session_state.people_list['Gender'])
    st.session_state.people_list['Category'] = standardize_string_column(st.session_state.people_list['Category'])

def load_image_base64(image_path):
    """
    Load an image file and return its base64 encoded string.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('UTF-8')
    except Exception as e:
        st.error(f"Could not load image at {image_path}: {e}")
        return None

def display_image_base64(image_base64, width_percent=100):
    """
    Return HTML code to display a base64-encoded image.
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
    """
    Select 20 questions probabilistically based on category and difficulty.
    """
    difficulty_patterns = {1: [0.6, 0.3, 0.1], 2: [0.4, 0.3, 0.3], 3: [0.2, 0.3, 0.5]}
    difficulty_distribution = difficulty_patterns[difficulty]
    selected_df = pd.DataFrame(columns=possible_questions.columns)
    current_round_questions = []

    if possible_questions.empty:
        st.error("No questions available to select from.")
        return asked_df, selected_df  # Return empty DataFrame

    # Work on a copy to avoid SettingWithCopyWarning
    possible_questions = possible_questions.copy()
    
    # Ensure correct data type for calculations
    possible_questions['Category_Prob'] = possible_questions['Category'].map(categories_share).astype(float)
    possible_questions['Level_Prob'] = possible_questions['Fame_Level'].map({
        1: difficulty_distribution[0],
        2: difficulty_distribution[1],
        3: difficulty_distribution[2]
    }).astype(float)
    
    possible_questions['Same_Cat_Level_Count'] = (
        possible_questions.groupby(['Category', 'Fame_Level'])['Name'].transform('count')
    ).astype(float)

    # Compute probabilities
    possible_questions['Prob'] = (
        possible_questions['Category_Prob'] * possible_questions['Level_Prob'] /
        possible_questions['Same_Cat_Level_Count']
    )

    # Handle NaN or zero probabilities
    possible_questions['Prob'] = possible_questions['Prob'].fillna(0)  # Replace NaN with 0
    prob_sum = possible_questions['Prob'].sum()

    if prob_sum == 0:
        st.error("No valid probabilities computed. Check data filtering.")
        return asked_df, selected_df  # Return empty DataFrame

    possible_questions['Prob'] /= prob_sum  # Normalize

    # Ensure dtype compatibility with np.random.choice
    possible_questions['Prob'] = possible_questions['Prob'].astype(float)

    # Pick a question based on probabilities
    chosen_row = np.random.choice(possible_questions.index, p=possible_questions['Prob'].values)
    chosen_question = possible_questions.loc[chosen_row]

    # Add chosen question to selected_df and remove from possible_questions
    selected_df = pd.concat([selected_df, chosen_question.to_frame().T], ignore_index=True)
    current_round_questions.append(chosen_question['Name'])
    asked_df = pd.concat([asked_df, chosen_question.to_frame().T], ignore_index=True)
    possible_questions = possible_questions.drop(chosen_row).reset_index(drop=True)
    
    return asked_df, selected_df

def generate_dummy_answers(selected_df, people_list, difficulty):
    """
    For each selected question, generate three dummy answers based on difficulty.
    """
    difficulty_patterns = {
        1: [0.6, 0.3, 0.1],
        2: [0.4, 0.3, 0.3],
        3: [0.2, 0.3, 0.5]
    }
    difficulty_distribution = difficulty_patterns[difficulty]
    dummy_answers = {}
    people_list = people_list[~people_list['Name'].isin(selected_df['Name'])]
    unique_categories = people_list['Category'].unique()
    unique_categories_count = len(unique_categories)
    
    for _, question in selected_df.iterrows():
        candidates = people_list[people_list['Gender'] == question['Gender'].lower()]
        category_probs = np.where(candidates['Category'] == question['Category'].lower(), 0.7, 0.3 / (unique_categories_count - 1))
        level_probs = np.array([difficulty_distribution[level - 1] for level in candidates['Fame_Level']])
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

    checkbox_dict = {category: st.session_state[category] for category in st.session_state.people_list['Category'].unique().tolist()}
        
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

    st.session_state.all_questions = st.session_state.people_list[st.session_state.people_list['Category'].isin(selected_categories)]
    st.session_state.possible_questions = st.session_state.all_questions[~st.session_state.all_questions['Name'].isin(st.session_state.asked_df['Name'])]
    category_counts = st.session_state.all_questions['Category'].value_counts()
    total_counts = category_counts.sum()
    categories_share = (category_counts / total_counts).to_dict()

    st.session_state.asked_df, st.session_state.selected_df = probabilistic_round_selection(
        st.session_state.possible_questions, 
        st.session_state.asked_df, 
        categories_share,
        selected_difficulty
    )

    st.session_state.dummy_answers = generate_dummy_answers(st.session_state.selected_df, st.session_state.people_list, selected_difficulty)

    st.session_state.selected_df['Image_Path'] = 'DATA/Pictures/' + st.session_state.selected_df['Name'].str.replace(' ', '_') + '.png'
    st.session_state.selected_df['Image_Base64'] = st.session_state.selected_df['Image_Path'].apply(load_image_base64)
    
    st.session_state.answers = {}
    st.session_state.state = 2
    st.session_state.counter = 0
    st.session_state.correct_answers = 0

def display_form():
    st.title("Face Quiz", anchor=False)
    st.write("\n")
    unique_categories = st.session_state.people_list['Category'].unique().tolist()

    with st.form(key='my_form'):
        checkbox_dict = {category: st.checkbox(category, key=category, value=True) for category in unique_categories}
        difficulty_levels = list(difficulty_map.keys())
        selected_difficulty = st.selectbox('Select Difficulty Level:', difficulty_levels, key="selected_difficulty", index=1)
        submit_button = st.form_submit_button(label='Start Quiz', on_click=start_quiz)

def submit_answer(answer):
    elapsed_time = time.time() - st.session_state.question_start_times[st.session_state.counter]
    is_correct = answer == st.session_state.selected_df.iloc[st.session_state.counter]['Name']
    base_score = round(BASE_SCORES[st.session_state.selected_df.iloc[st.session_state.counter]['Fame_Level']]) if is_correct else 0
    time_bonus = round(calculate_exponential_time_bonus(elapsed_time)) if is_correct else 0
    total_score = round(st.session_state.multiplier * (base_score + time_bonus)) if is_correct else 0
    
    st.session_state.score_breakdown[st.session_state.counter] = {
        "base_score": round(base_score),
        "time_bonus": round(time_bonus),
        "multiplier": round(st.session_state.multiplier, 2),
        "total_score": round(total_score),
        "elapsed_time": round(elapsed_time, 2)
    }

    update_multiplier(is_correct)
    st.session_state.score += round(total_score)
    st.session_state.answers[st.session_state.counter] = answer

    if st.session_state.counter < 19: 
        st.session_state.counter += 1
    else:
        st.session_state.state = 3

def display_question():
    st.empty()

    # Check if selected_df exists and has rows
    if "selected_df" not in st.session_state or st.session_state.selected_df.empty:
        st.error("No questions available. Please restart the quiz.")
        return

    # Ensure the counter is within bounds
    if st.session_state.counter >= len(st.session_state.selected_df):
        st.error("No more questions left. Restarting...")
        st.session_state.state = 3  # Move to results state
        return

    # Get the current question safely
    question = st.session_state.selected_df.iloc[st.session_state.counter]

    col1, col2, col3 = st.columns([7,3,3])
    col2.markdown(format_title("Score:", align='right'), unsafe_allow_html=True)
    score_placeholder = col3.empty()
    score_placeholder.markdown(format_title(st.session_state.display_score, align='right'), unsafe_allow_html=True)
    progress_bar = st.empty()
    progress_bar.progress(1.0) 

    col1.markdown(format_title(f"{st.session_state.counter+1}/20"), unsafe_allow_html=True)
    
    cols = st.columns([5,5])
    cols[0].markdown(display_image_base64(question['Image_Base64'], 70), unsafe_allow_html=True)

    options = [question['Name']] + st.session_state.dummy_answers[question['Name']]
    random.shuffle(options)

    if "question_start_times" not in st.session_state:
        st.session_state.question_start_times = {}

    if st.session_state.counter not in st.session_state.question_start_times:
        st.session_state.question_start_times[st.session_state.counter] = time.time()
        st.session_state.t_end = st.session_state.question_start_times[st.session_state.counter] + 1

    cols[1].subheader("Who is this person?", anchor=False)
    cols[1].markdown("\n")
    colz = cols[1].columns(2)

    for i in range(4):
        colz[(i // 2)].button(options[i], key=f"option_{i+1}", on_click=submit_answer, args=(options[i],), use_container_width=True)

    st.markdown(f"{question['Short_Description']}")

    while True:
        elapsed_time = time.time() - st.session_state.question_start_times[st.session_state.counter]
        progress_value = max(0.0, min(1.0, 1.0 - (elapsed_time / QUESTION_DURATION)))
        progress_bar.progress(progress_value)
        remaining_frames = (QUESTION_DURATION - elapsed_time) / REFRESH_RATE
        total_frames = QUESTION_DURATION / REFRESH_RATE
        st.session_state.display_score = compute_next_display_score(
            st.session_state.display_score,
            st.session_state.score,
            remaining_frames,
            total_frames,
            lambda_=5
        )
        score_placeholder.markdown(format_title(st.session_state.display_score, align='right'), unsafe_allow_html=True)
        
        if elapsed_time > QUESTION_DURATION:
            submit_answer("No Answer")
            st.experimental_rerun()
        elif st.session_state.counter in st.session_state.answers:
            st.experimental_rerun()
        
        time.sleep(REFRESH_RATE)


def display_results():
    st.empty()
    if st.session_state.correct_answers == 20:
        st.balloons()
        st.subheader("Congratulations! You answered **all** 20 questions correctly.", anchor=False)
    else:
        st.write(f"You answered {st.session_state.correct_answers} out of 20 questions correctly ({int(st.session_state.correct_answers/20*100)}%).")

    col1, colx, col2, col3 = st.columns([3, 3, 5, 3])
    col1.markdown("\n")
    new_game_button = col1.button('New game', on_click=new_game, use_container_width=True)
    col2.markdown(format_title("Your score:", align='right'), unsafe_allow_html=True)
    score = st.session_state.score
    formatted_score = "{:,}".format(score).replace(",", " ")
    col3.markdown(format_title(formatted_score, align='right'), unsafe_allow_html=True)

    for i in range(20):
        st.write("-----")
        col1, col2, col3 = st.columns([3,8,3])
        image = st.session_state.selected_df.iloc[i]['Image_Base64']
        name = st.session_state.selected_df.iloc[i]['Name']
        text = st.session_state.selected_df.iloc[i]['Short_Description']
        breakdown = st.session_state.score_breakdown[i]
        col1.markdown(display_image_base64(image, 90), unsafe_allow_html=True)
        if breakdown['total_score'] != 0:
            col2.subheader(f":white_check_mark: {name}", anchor=False)
            helper = f"(Question: **{breakdown['base_score']}** + Speed bonus: **{breakdown['time_bonus']}**) x Streak multiplier: **{breakdown['multiplier']}**"
            col3.subheader(f"**{breakdown['total_score']}**", anchor=False, help=helper)
            col3.write(f"{breakdown['elapsed_time']} seconds")
        else:
            col2.subheader(f":x: {name}", anchor=False)
            helper = f"~~{st.session_state.answers[i]}~~"
            col3.subheader(f"**{breakdown['total_score']}**", anchor=False)
            col3.write(helper)
        col2.write(f"{text}")

def reset():
    st.session_state.clear()


##### FOR DEBUG PURPOSES
# # import json
# # # Convert session state to a serializable format
# # def serialize_session_state(state):
# #     serializable_state = {}
# #     for key, value in state.items():
# #         if isinstance(value, pd.DataFrame):
# #             # Convert DataFrame to CSV string
# #             serializable_state[key] = {
# #                 "__type__": "DataFrame",
# #                 "data": value.to_csv(index=False)
# #             }
# #         else:
# #             serializable_state[key] = value
# #     return serializable_state

# Main function
def main():

    #st.empty()

    #### FOR DEBUG PURPOSES
    # # Reset button
    # #reset_button = st.button('Reset Game', on_click=reset)

    # # st.session_state
    
    # # # Save the serializable session state to a JSON file
    # # serialized_state = serialize_session_state(st.session_state)
    # # with open('session_state.json', 'w') as f:
    # #     json.dump(serialized_state, f)

    # # # Create a download button in Streamlit for the JSON file
    # # st.download_button(
    # #     label="Download session state",
    # #     data=open('session_state.json', 'rb'),
    # #     file_name='session_state.json',
    # #     mime='application/json'
    # # )


    # Initialize session state variables
    if "people_list" not in st.session_state:
        init()

    if "asked_df" not in st.session_state:
        st.session_state.asked_df = pd.DataFrame(columns=st.session_state.people_list.columns)

    if "state" not in st.session_state:
        st.session_state.state = 1

    if "score_breakdown" not in st.session_state:
        st.session_state.score_breakdown = {}

    if "answer" not in st.session_state:
        st.session_state.answer = ""
    
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

if __name__ == "__main__":
    main()
