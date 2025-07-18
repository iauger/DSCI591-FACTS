import json
import urllib.request
from datasets import load_dataset, Dataset, DatasetDict
from typing import Optional, Union
import pandas as pd
from pathlib import Path

import requests
from tqdm import tqdm

class DataDownLoader:
    def __init__(self, output_path, force = False):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.force = force
    
    @staticmethod
    def _file_exists(path: Path, force: bool) -> bool:
        if path.exists():
            if force:
                tqdm.write(f"Overwriting existing file: {path}")
                return False
            else:
                tqdm.write(f"Skipping download, file already exists: {path}")
                return True
        return False
    
    def url_download(self, url: str, dataset_name: str) -> Optional[Path]:
        dest = self.output_path / dataset_name
        
        if self._file_exists(dest, self.force):
            return dest
        
        try: 
            urllib.request.urlretrieve(url, dest)
            tqdm.write(f"Data written to {dest}")
            return dest
        except Exception as e:
            tqdm.write(f"Failed to download {url}: {e}")
            return None
    
    def download_and_load_parquet(self, url: str, local_name: str, raw_dir: Path, to_csv: bool = True) -> pd.DataFrame:
        local_path = raw_dir / local_name
        try:
            if not local_path.exists():
                response = requests.get(url)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(response.content)
                tqdm.write(f"Downloaded Parquet: {local_path}")
            else:
                tqdm.write(f"Parquet already cached: {local_path}")
        except requests.exceptions.RequestException as e:
            tqdm.write(f"Failed to download Parquet from {url}: {e}")
            raise

        df = pd.read_parquet(local_path)
        tqdm.write(f"Loaded {len(df):,} rows from {local_path.name}")

        if to_csv:
            csv_path = local_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            tqdm.write(f"Converted Parquet to CSV: {csv_path}")

        return df
        
    def hugging_face_download(
    self,
    dataset_name: str,
    subset: Optional[str] = None,
    split: Union[str, list[str]] = "train",
    file_format: str = "json"
) -> list[Path]:
        try:
            # Load dataset (with or without subset)
            if subset:
                dataset = load_dataset(dataset_name, subset)
            else:
                dataset = load_dataset(dataset_name)

            # Get list of splits to download
            if isinstance(split, str):
                if split == "all":
                    if isinstance(dataset, DatasetDict):
                        splits = list(dataset.keys())  # ['train', 'test', ...]
                    else:
                        splits = ["train"]
                else:
                    splits = [split]
            else:
                splits = split

            saved_paths = []
            for s in splits:
                split_data = dataset[str(s)]

                if not isinstance(split_data, Dataset):
                    raise TypeError(f"Expected HuggingFace Dataset for split '{s}', got {type(split_data)}")

                ext = "csv" if file_format == "csv" else "json"
                output_path = self.output_path / f"{dataset_name}_{s}.{ext}"
                
                # Check if file already exists
                if self._file_exists(output_path, self.force):
                    saved_paths.append(output_path)
                    continue
                
                if file_format == "csv":
                    split_data.to_csv(str(output_path))
                else:
                    split_data.to_json(str(output_path))

                tqdm.write(f"Saved {s} split to {output_path}")
                saved_paths.append(output_path)

            return saved_paths

        except Exception as e:
            tqdm.write(f"Failed to download dataset '{dataset_name}': {e}")
            return []

        
        
    def download_json(self, dataset_name: str, url: str):
        return self.url_download(url, dataset_name + '.json')
    
    def download_csv(self, dataset_name: str, url: str):
        return self.url_download(url, dataset_name + '.csv')
    
    

    
    
        