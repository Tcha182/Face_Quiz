import streamlit as st
import pandas as pd
import numpy as np
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

def new_game():
    st.session_state.state = 1


def updated_probabilistic_round_selection(current_round_questions, all_questions, asked_df, categories_share, difficulty):
    difficulty_patterns = {1: [0.6, 0.3, 0.1], 2: [0.4, 0.3, 0.3], 3: [0.2, 0.3, 0.5]}
    difficulty_distribution = difficulty_patterns[difficulty]

    selected_indices = []

    for _ in range(20):
        # Compute probabilities for each question
        current_round_questions['Category_Prob'] = current_round_questions['Category'].map(categories_share)
        current_round_questions['Level_Prob'] = current_round_questions['Fame_Level'].map({1: difficulty_distribution[0], 2: difficulty_distribution[1], 3: difficulty_distribution[2]})
        current_round_questions['Same_Cat_Level_Count'] = current_round_questions.groupby(['Category', 'Fame_Level'])['Name'].transform('count')
        current_round_questions['Prob'] = current_round_questions['Category_Prob'] * current_round_questions['Level_Prob'] / current_round_questions['Same_Cat_Level_Count']

        # Normalize the probabilities
        current_round_questions['Prob'] = current_round_questions['Prob'] / current_round_questions['Prob'].sum()

        # Check if sum of probabilities is zero
        if current_round_questions['Prob'].sum() == 0:
            # Rollback all questions from asked_df excluding the ones selected in the current round
            current_round_questions = pd.concat([current_round_questions, asked_df[~asked_df.index.isin(selected_indices)]])
            asked_df = asked_df[asked_df.index.isin(selected_indices)]

            # Recalculate the probabilities
            current_round_questions['Category_Prob'] = current_round_questions['Category'].map(categories_share)
            current_round_questions['Level_Prob'] = current_round_questions['Fame_Level'].map({1: difficulty_distribution[0], 2: difficulty_distribution[1], 3: difficulty_distribution[2]})
            current_round_questions['Same_Cat_Level_Count'] = current_round_questions.groupby(['Category', 'Fame_Level'])['Name'].transform('count')
            current_round_questions['Prob'] = current_round_questions['Category_Prob'] * current_round_questions['Level_Prob'] / current_round_questions['Same_Cat_Level_Count']
            current_round_questions['Prob'] = current_round_questions['Prob'] / current_round_questions['Prob'].sum()

        # Select a question based on their probabilities
        selected_idx = current_round_questions.sample(n=1, weights='Prob').index[0]
        selected_indices.append(selected_idx)
        
        selected_question = current_round_questions.loc[selected_idx]

        # Drop the selected question from the current round questions
        current_round_questions = current_round_questions.drop(selected_idx)
        
        # Check if the Category and Fame_Level combination of the selected question is exhausted and rollback if necessary
        if not current_round_questions[(current_round_questions['Category'] == selected_question['Category']) & (current_round_questions['Fame_Level'] == selected_question['Fame_Level'])].shape[0]:
            to_rollback = asked_df[(asked_df['Category'] == selected_question['Category']) & (asked_df['Fame_Level'] == selected_question['Fame_Level']) & (~asked_df.index.isin(selected_indices))]
            current_round_questions = pd.concat([current_round_questions, to_rollback])
            asked_df = asked_df.drop(to_rollback.index)

    # Separate out the selected questions and update the possible and asked questions
    selected_df = all_questions.loc[selected_indices]
    
    # Add the selected questions to the asked_df
    asked_df = pd.concat([asked_df, selected_df])

    return current_round_questions, asked_df, selected_df


def generate_dummy_answers(selected_df, people_list, difficulty):
    # Define the difficulty patterns as in the probabilistic_round_selection function
    difficulty_patterns = {
        1: [0.6, 0.3, 0.1],
        2: [0.4, 0.3, 0.3],
        3: [0.2, 0.3, 0.5]
    }
    difficulty_distribution = difficulty_patterns[difficulty]
    
    # Remove the questions of the current round from people_list
    people_list = people_list[~people_list['Name'].isin(selected_df['Name'])]
    
    # Dictionary to store dummy answers for each question
    dummy_answers = {}
    
    for _, question in selected_df.iterrows():
        # Filter candidates with the same gender as the current question
        candidates = people_list[people_list['Gender'] == question['Gender']]
        
        probabilities = []
        
        for _, candidate in candidates.iterrows():
            # Calculate category probability
            if candidate['Category'] == question['Category']:
                category_prob = 0.7
            else:
                category_prob = 0.3 / (len(candidates['Category'].unique()) - 1)
            
            # Calculate fame level probability
            level_prob = difficulty_distribution[candidate['Fame_Level'] - 1]
            
            # Multiply category and fame level probabilities
            total_prob = category_prob * level_prob
            probabilities.append(total_prob)
        
        # Normalize the probabilities
        probabilities = [prob / sum(probabilities) for prob in probabilities]
        
        # Randomly select three dummy answers
        dummy_names = np.random.choice(candidates['Name'], size=3, p=probabilities, replace=False)
        
        dummy_answers[question['Name']] = list(dummy_names)
        
    return dummy_answers


def start_quiz():
    # 1. Retrieve the categories the user has selected.
    checkbox_dict = {category: st.session_state[category] for category in st.session_state.people_list['Category'].unique().tolist()}
    selected_difficulty = st.session_state.selected_difficulty
    selected_categories = [key for key, value in checkbox_dict.items() if value]

    # 2. Create the categories_share dictionary based on the entire dataset of the selected categories.
    all_selected_questions = st.session_state.people_list[st.session_state.people_list['Category'].isin(selected_categories)]
    category_counts = all_selected_questions['Category'].value_counts()
    total_counts = category_counts.sum()
    categories_share = (category_counts / total_counts).to_dict()

    # 3. Define current_round_questions from possible_questions filtered by the user-selected categories.
    current_round_questions = st.session_state.possible_questions[st.session_state.possible_questions['Category'].isin(selected_categories)]

    # 4. Call the improved_probabilistic_round_selection function using current_round_questions to select questions for the current round.
    st.session_state.possible_questions, st.session_state.asked_df, st.session_state.selected_df = probabilistic_round_selection(
        current_round_questions, 
        st.session_state.people_list,
        st.session_state.asked_df, 
        categories_share, 
        selected_difficulty)

    # 5. Generate dummy answers.
    st.session_state.dummy_answers = generate_dummy_answers(st.session_state.selected_df, st.session_state.people_list, selected_difficulty)

    # 6. Load images for the selected questions.
    st.session_state.selected_df['Image_Path'] = 'DATA/Pictures/' + st.session_state.selected_df['Name'].str.replace(' ', '_') + '.png'
    st.session_state.selected_df['Image'] = st.session_state.selected_df['Image_Path'].apply(load_image)

    # 7. Initialize other session state variables for the quiz.
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
        selected_difficulty = st.selectbox('Select Difficulty Level:', difficulty_levels, key="selected_difficulty")
        
        # Start Quiz button
        submit_button = st.form_submit_button(label='Start Quiz', 
                                  on_click=start_quiz)

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
    
    # Define possible answers
    options = [question['Name']] + st.session_state.dummy_answers[question['Name']]
    random.shuffle(options)
    
    # Get the answer from the user
    cols = st.columns(2)
    for i in range(4):
        cols[i//2].button(options[i], key=f"option_{i+1}", on_click=submit_answer, args=(options[i],))

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

    new_game_button = st.button('New game', on_click=new_game)

# Function to handle reset button click
def reset():
    st.session_state.clear()


# Main function
def main():

    st.session_state
    try: st.dataframe(st.session_state.asked_df)
    except:
        pass

    # Initialize session state variables
    if "people_list" not in st.session_state:
        init()

    if "asked_df" not in st.session_state:
        st.session_state.asked_df = pd.DataFrame(columns=st.session_state.people_list.columns)

    if "possible_questions" not in st.session_state:
        st.session_state.possible_questions = st.session_state.people_list.copy()

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