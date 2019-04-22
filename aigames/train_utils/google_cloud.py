from google.cloud import storage


def google_cloud_upload_file(job_dir, filename):
    # Taken from google cloud hyperparameter tuning example
    """Saves the model to Google Cloud Storage"""
    # Example: job_dir = 'gs://BUCKET_ID/hptuning_sonar/1'
    job_dir = job_dir.replace('gs://', '')  # Remove the 'gs://'
    # Get the Bucket Id
    bucket_id = job_dir.split('/')[0]
    # Get the path. Example: 'hptuning_sonar/1'
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))

    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob(f'{bucket_path}/{filename}')
    blob.upload_from_filename(filename)
