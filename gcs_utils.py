import streamlit as st
import os
from google.oauth2 import service_account
from google.cloud import storage
from io import BytesIO
import json

# Initialize Google Cloud Storage client using environment credentials
def initialize_client():
    credentials_json = st.secrets["GS_CREDENTIALS"]["google_credentials_json"]
    credentials = service_account.Credentials.from_service_account_info(json.loads(credentials_json))
    return storage.Client(credentials=credentials)

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name, timeout=600):
    """Uploads a file to the GCS bucket with an adjustable timeout."""
    client = initialize_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Setting a higher timeout to handle larger files
    blob.upload_from_filename(source_file_name, timeout=timeout)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def download_from_gcs(bucket_name, blob_name):
    """Downloads a file from GCS and returns it as a BytesIO object."""
    client = initialize_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    data = BytesIO()
    blob.download_to_file(data)
    data.seek(0)  # Rewind the file pointer for reading
    print(f"File {blob_name} downloaded from GCS.")
    return data

def file_exists_in_gcs(bucket_name, blob_name):
    """Checks if a file exists in GCS."""
    client = initialize_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()

def list_files_in_gcs(bucket_name, prefix=None):
    """Lists all files in a given bucket with an optional prefix."""
    client = initialize_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]
