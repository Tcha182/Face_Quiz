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
- **Create your own category**: type any topic (e.g. *tennis players*) and the app builds a brand-new quiz category on the fly — an LLM generates the list of people with descriptions, their portraits are fetched from Wikipedia (with a Wikidata fallback), and OpenCV detects the faces and produces circular portraits in the standard format before everything is saved to Google Cloud Storage.
- **Category management**: a password-protected Manage tab shows every category with its provenance (original vs. user-created) and creation date, and lets the admin delete categories, cleaning up user-added pictures from storage.

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
- google-cloud-storage
- openai
- requests
- opencv-python-headless
- Pillow

## Configuration

The app expects the following Streamlit secrets (`.streamlit/secrets.toml`):

```toml
OPENAI_API_KEY = "sk-..."       # used by the "Create a category" feature
# OPENAI_MODEL = "gpt-4o-mini"  # optional, defaults to gpt-4o-mini
ADMIN_PASSWORD = "..."          # unlocks the Manage tab (category deletion)

[GS_CREDENTIALS]
GS_BUCKET_NAME = "your-bucket"
google_credentials_json = '{...service account JSON...}'
```

## Technical Details

- **Streamlit**: The UI is built using Streamlit, for simplicity.
- **LLM integration**: The lists of famous persons and their descriptions are generated with an OpenAI model (gpt-4o-mini by default) — both for the original dataset and live, whenever a user creates a new category from the app.
- **Wikipedia**: Portraits are fetched from Wikipedia through the MediaWiki API to ensure that they are free to use.
- **Image Processing**: OpenCV is used for face detection and image normalization: every portrait is centered on the face, cropped to a 500×500 square, masked to a circle with a grey ring border, and saved as a 520×520 RGBA PNG.
- **Google Cloud Storage**: Both the people list (CSV) and the processed pictures are stored in a GCS bucket.

## License

This project is licensed under the MIT License.
