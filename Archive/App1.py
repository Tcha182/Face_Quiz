import streamlit as st
import pandas as pd
from itertools import repeat
import random
from PIL import Image


def init(start=False):
    load_table()

def load_table():
    st.session_state.people_list = pd.read_excel("DATA/People_list.xlsx")

def load_image(image_path):
    return Image.open(image_path)

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


def display_form():
    unique_categories = st.session_state.people_list['Category'].unique().tolist()

    with st.form(key='my_form'):
        checkbox_dict = {category: st.checkbox(category, key=category, value=True) for category in unique_categories}
        
        difficulty_levels = [1, 2, 3]
        selected_difficulty = st.selectbox('Select Difficulty Level:', difficulty_levels)
        
        submit_button = st.form_submit_button(label='Start Quiz')

        if submit_button:
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
            
            st.session_state.user_answers = {}
            st.session_state.state = 2
            st.session_state.counter = 0
            st.session_state.correct_answers = 0


def display_question(counter):
    question = st.session_state.selected_df.iloc[counter]
    st.write(f"Question {counter+1}/20: What is the name of this person?")
    st.image(question['Image']) 
    answer = st.text_input('Your answer:')
    
    if st.button("Submit Answer"):
        st.session_state.answer = answer
        if 'answer' in st.session_state and st.session_state.answer == st.session_state.selected_df.iloc[st.session_state.counter]['Name']:
            st.session_state.correct_answers += 1
            
        # store the user's answer and correct answer in the dictionary
        st.session_state.user_answers[counter] = (st.session_state.answer, st.session_state.selected_df.iloc[st.session_state.counter]['Name'])
        
        st.session_state.answer = ""  # clear the answer field for the next question

        if st.session_state.counter < 19: 
            st.session_state.counter += 1
        else:
            st.session_state.state = 3


def display_results():
    st.write(f"You answered {st.session_state.correct_answers} questions correctly out of 20.")
    st.write(f"Your success rate is {st.session_state.correct_answers/20*100}%.")
    
    # display the answers
    for i in range(20):
        st.write(f"Question {i+1}: Your answer was '{st.session_state.user_answers[i][0]}'. The correct answer was '{st.session_state.user_answers[i][1]}'.")

    if st.button("Start New Series"):
        st.session_state.state = 1


def main():
    if "people_list" not in st.session_state:
        init()

    if "state" not in st.session_state:
        st.session_state.state = 1

    if st.session_state.state == 1:
        display_form()

    elif st.session_state.state == 2:
        display_question(st.session_state.counter)

    elif st.session_state.state == 3:
        display_results()

    if st.button('Reset'):
        st.session_state.clear()


if __name__ == '__main__':
    main()