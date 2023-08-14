import streamlit as st
import pandas as pd
from itertools import repeat
import random
from PIL import Image
import time

# Function to initialize the app by loading the list of people
def init():
    load_table()

# Function to load the excel file containing the list of people
def load_table():
    st.session_state.people_list = pd.read_excel("DATA/People_list.xlsx")

# Function to load an image from a given path
def load_image(image_path):
    try:
        return Image.open(image_path)
    except (IOError, SyntaxError) as e:
        st.error(f"Could not load image at {image_path}: {e}")
        return None

# Function to select a round of questions based on difficulty
def round_selection(possible_questions, asked_df, unique_categories, categories_share, difficulty=1):
    patterns =  [[12,6,2],[8,8,4],[4,10,6]]
    data = [1,2,3]
    if difficulty in [1, 2, 3]:
        pattern = patterns[difficulty - 1]
        question_level = [val for val, count in zip(data, pattern) for _ in repeat(val, count)]
        random.shuffle(question_level)

    selected_category = []
    for i in range(len(question_level)):
        selected_category.append(random.choices(unique_categories, categories_share)[0])

    df = pd.DataFrame({'selected_category': selected_category, 'question_level': question_level})

    selected_df = pd.DataFrame()
    current_round_questions = pd.DataFrame() # Questions asked in the current round

    for idx, row in df.iterrows():
        matching_rows = possible_questions[(possible_questions['Category'] == row['selected_category']) & 
                                    (possible_questions['Fame_Level'] == row['question_level'])]

        if matching_rows.empty:
            asked_from_category = asked_df[(asked_df['Category'] == row['selected_category']) & 
                                            (asked_df['Fame_Level'] == row['question_level'])]
            possible_questions = pd.concat([possible_questions, asked_from_category])
            asked_df = asked_df.drop(asked_from_category.index)

            matching_rows = possible_questions[(possible_questions['Category'] == row['selected_category']) & 
                                        (possible_questions['Fame_Level'] == row['question_level'])]

        if not matching_rows.empty:
            chosen_row = matching_rows.sample(n=1)

            current_round_questions = pd.concat([current_round_questions, chosen_row])
            selected_df = selected_df.append(chosen_row)

            asked_df = pd.concat([asked_df, chosen_row])
            possible_questions = possible_questions.drop(chosen_row.index)

    return possible_questions, asked_df, selected_df

def start_quiz(checkbox_dict, selected_difficulty):
    selected_categories = [key for key, value in checkbox_dict.items() if value]
    
    st.session_state.all_questions = st.session_state.people_list[st.session_state.people_list['Category'].isin(selected_categories)]
    
    category_counts = st.session_state.all_questions['Category'].value_counts()
    total_counts = category_counts.sum()
    unique_categories = category_counts.index.tolist()
    categories_share = (category_counts / total_counts).tolist()
    
    asked_df = pd.DataFrame(columns=st.session_state.all_questions.columns)
    
    st.session_state.possible_questions, st.session_state.asked_df, st.session_state.selected_df = round_selection(st.session_state.all_questions, 
                                                                        asked_df, 
                                                                        unique_categories, 
                                                                        categories_share, 
                                                                        selected_difficulty)

    st.session_state.selected_df['Image_Path'] = 'DATA/Pictures/' + st.session_state.selected_df['Name'].str.replace(' ', '_') + '.png'
    st.session_state.selected_df['Image'] = st.session_state.selected_df['Image_Path'].apply(load_image)
    
    st.session_state.answers = {}
    st.session_state.state = 2
    st.session_state.counter = 0
    st.session_state.correct_answers = 0



# Function to display the form for starting the quiz
def display_form():
    unique_categories = st.session_state.people_list['Category'].unique().tolist()

    with st.form(key='my_form'):
        # Display checkboxes for category selection
        checkbox_dict = {category: st.checkbox(category, key=category, value=True) for category in unique_categories}
        
        # Display dropdown for difficulty selection
        difficulty_levels = [1, 2, 3]
        selected_difficulty = st.selectbox('Select Difficulty Level:', difficulty_levels)
        
        # Start Quiz button
        submit_button = st.form_submit_button(label='Start Quiz', 
                                  on_click=start_quiz,
                                  args=(checkbox_dict, selected_difficulty))

# Function to handle submit answer button click
def submit_answer(answer):
    # If answer is correct, increment correct_answers
    if answer == st.session_state.selected_df.iloc[st.session_state.counter]['Name']:
        st.session_state.correct_answers += 1
                
    # Store the answer
    st.session_state.answers[st.session_state.counter] = answer

    # Move to the next question or end the quiz if it was the last question
    if st.session_state.counter < 19: 
        st.session_state.counter += 1
    else:
        st.session_state.state = 3

# Function to display the current question
def display_question():
    # Get the current question
    question = st.session_state.selected_df.iloc[st.session_state.counter]
    
    # Display the question
    st.write(f"Question {st.session_state.counter+1}/20: What is the name of this person?")
    st.image(question['Image'])
    
    # Get the answer from the user
    answer = st.text_input('Your answer:', value=st.session_state.answer)
    
    # Submit answer button
    submit_button = st.button("Submit Answer", on_click=submit_answer, args=(answer,))

# Function to display the quiz results
def display_results():
    # Display correct answer count and success rate
    st.write(f"You answered {st.session_state.correct_answers} questions correctly out of 20.")
    st.write(f"Your success rate is {st.session_state.correct_answers/20*100}%.")

    # Display each question, the user's answer, and the correct answer
    for i in range(20):
        st.write(f"Question {i+1}:")
        st.write(f"Your answer: {st.session_state.answers[i]}")
        st.write(f"Correct answer: {st.session_state.selected_df.iloc[i]['Name']}")

# Function to handle reset button click
def reset():
    st.session_state.clear()

# Main function
def main():

    st.session_state

    # Initialize session state variables
    if "people_list" not in st.session_state:
        init()

    if "state" not in st.session_state:
        st.session_state.state = 1

    if "answer" not in st.session_state:
        st.session_state.answer = ""

    if st.session_state.state == 1:
        # State 1: Initial State
        display_form()

    elif st.session_state.state == 2:
        # State 2: Question State
        display_question()

    elif st.session_state.state == 3:
        # State 3: Finish State
        display_results()

    # Reset button
    reset_button = st.button('Reset', on_click=reset)

# Call the main function
if __name__ == "__main__":
    main()