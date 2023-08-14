import pandas as pd
import streamlit as st
from PIL import Image

# List of picture and text combinations
data = pd.read_excel("DATA/People_list.xlsx")
data = data[:20]

# Function to load an image from a given path
def load_image(image_path):
    try:
        return Image.open(image_path)
    except (IOError, SyntaxError) as e:
        st.error(f"Could not load image at {image_path}: {e}")
        return None

data['Image_Path'] = 'DATA/Pictures/' + data['Name'].str.replace(' ', '_') + '.png'
data["Image"] = data['Image_Path'].apply(load_image)

# Set the number of columns and rows
num_columns = 5
num_rows = 4

# Calculate the total number of elements
total_elements = num_columns * num_rows

# Ensure that we have enough data for the grid
if len(data) < total_elements:
    st.warning(f"Insufficient data. Need at least {total_elements} items.")
    data = data.append([{"image_path": "", "text": ""}] * (total_elements - len(data)), ignore_index=True)

st.title("Picture and Text Grid")

# Create the grid layout
for row in range(num_rows):
    cols = st.columns(num_columns)
    for col_index, col in enumerate(cols):
        item_index = row * num_columns + col_index

        image = data["Image"].iloc[item_index]
        text = data["Name"].iloc[item_index]
        if image:
            b = col.title("2023", anchor=False)
            col.image(image, use_column_width=True)
            a = col.button(label=f"**:green[{item_index+1}]: {text}**")

        else:
            col.empty()

col1, col3 = st.columns([10 ,3])

col1.markdown("<h1 style='text-align: right; ;'>Your score:</h1>", unsafe_allow_html=True)
col3.title("34529", anchor=False)

# Create the grid layout
for row in range(len(data)):
    col1, col2, col3 = st.columns([2,8,3])
    image = data["Image"].iloc[row]
    name = data["Name"].iloc[row]
    text = data["Short_Description"].iloc[row]

    col1.image(image, use_column_width=True)
    col2.subheader(f"{name}",anchor=False)
    col2.write(f"{text}")

    col3.subheader(f"**2023**",anchor=False)
    col3.write("a + b x ccccc")

