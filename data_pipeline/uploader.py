from typing import Optional
from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound
from google.auth import default
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

class DataUploader:
    def __init__(
        self, 
        bq_client: Optional[bigquery.Client] = None, 
        storage_client: Optional[storage.Client] = None, 
        project_id: Optional[str] = None, 
        dataset_name: Optional[str] = None, 
        bucket_name: Optional[str] = None
        ):
        self.bq_client = bq_client
        self.storage_client = storage_client
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.bucket_name = bucket_name
    
    def upload_to_bigquery(self, file_path: Path, table_name: str, write_mode = "WRITE_TRUNCATE") -> bool:
        """
        Uploads a file to BigQuery.
        
        Args:
            file_path (Path): Path to the file to upload.
            table_name (str): Name of the BigQuery table.
            write_mode (str): Write mode for BigQuery (default is "WRITE_TRUNCATE").
        
        Returns:
            bool: True if upload was successful, False otherwise.
        """
        if not file_path.exists():
            tqdm.write(f"File {file_path} does not exist. Skipping upload to BigQuery.")
            return False
        
        if file_path.suffix not in ['.csv', '.jsonl']:
            tqdm.write(f"Unsupported file format {file_path.suffix}. Only CSV and JSONL are supported.")
            return False
        
        safe_table_name = table_name.replace("-", "_").replace(".", "_")

        table_id = f"{self.project_id}.{self.dataset_name}.{safe_table_name}"
        
        try:
            if self.bq_client is None:
                tqdm.write("BigQuery client is not initialized. Skipping upload to BigQuery.")
                return False
            self.bq_client.get_table(table_id)
            tqdm.write(f"Table {table_id} already exists. Using existing table.")
        except NotFound:
            tqdm.write(f"Creating new table {table_id}.") 
            
            
        source_format = (
            bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
            if file_path.suffix == '.jsonl' 
            else bigquery.SourceFormat.CSV
        )
        
        job_config = bigquery.LoadJobConfig(
            source_format=source_format,
            autodetect=True,
            write_disposition=write_mode
        )

        try:
            with open(file_path, "rb") as f:
                if self.bq_client is None:
                    tqdm.write("BigQuery client is not initialized. Skipping upload to BigQuery.")
                    return False
                job = self.bq_client.load_table_from_file(f, table_id, job_config=job_config)
            job.result()  # Wait for the job to complete
            tqdm.write(f"Successfully uploaded {file_path} to BigQuery table {table_id}.")
            return True
        except Exception as e:
            tqdm.write(f"Failed to upload {file_path} to BigQuery: {e}")
            return False
    
    def upload_to_gcs(self, file_path: Path, destination_blob_name: str) -> bool:
        """
        Uploads a file to Google Cloud Storage.
        
        Args:
            file_path (Path): Path to the file to upload.
            destination_blob_name (str): Destination blob name in GCS.
        
        Returns:
            bool: True if upload was successful, False otherwise.
        """
        if not file_path.exists():
            tqdm.write(f"File {file_path} does not exist. Skipping upload to GCS.")
            return False

        if self.storage_client is None:
            tqdm.write("Storage client is not initialized. Skipping upload to GCS.")
            return False
        
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        try:
            blob.upload_from_filename(file_path)
            tqdm.write(f"Successfully uploaded {file_path} to GCS bucket {self.bucket_name} as {destination_blob_name}.")
            return True
        except Exception as e:
            tqdm.write(f"Failed to upload {file_path} to GCS: {e}")
            return False