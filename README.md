# Face Quiz

This application is a game where you gain points by recognizing famous persons. It was developed as a learning project to explore and utilize various tools and technologies in Python.

## Learning Objectives

- **Streamlit**: Utilized for creating a dynamic and interactive user interface.
- **GPT-4o-mini**: Used for generating the list of persons and small descriptions for each person.
- **Web Scraping**: Employed to gather pictures from Wikipedia for the listed persons.
- **Image Processing**: Applied facial recognition algorithms to normalize and standardize all pictures.

## Features

- Probabilistic question-picking engine, minimizing recurring questions on consecutive runs while ensuring consistent distribution of categories and difficulty levels.
- Dynamic and interactive quiz interface.
- Scoring based on accuracy and response time.
- Streaks and multipliers to enhance gameplay.
- Review of correct and incorrect answers at the end of the quiz.

## Usage

The app is available here: [Face Quiz](https://facequiz.streamlit.app/)

## Example

1. Start the quiz by selecting categories and difficulty level.
2. Recognize the famous faces presented.
3. Gain points based on accuracy and speed.
4. Review your score and correct answers at the end of the quiz.

## Dependencies

- streamlit
- pandas
- numpy
- random
- time
- math
- base64
- OpenCV
- OpenAI (API integration)

## Technical Details

- **Streamlit**: The UI is built using Streamlit, for simplicity.
- **GPT-4o-mini**: The list of famous persons and their descriptions were generated using GPT, demonstrating the integration of AI-generated content.
- **Web Scraping**: Pictures of famous persons were scraped from Wikipedia to ensure that they are free to uset.
- **Image Processing**: OpenCV was used for facial recognition and image normalization, ensuring that all pictures are centered on the face and standardized.

## License

This project is licensed under the MIT License.
