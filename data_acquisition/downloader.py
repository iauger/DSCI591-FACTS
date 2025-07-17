import json
import urllib.request
from datasets import load_dataset, Dataset, DatasetDict
from typing import Optional, Union
import pandas as pd
from pathlib import Path

from tqdm import tqdm

class DataDownLoader:
    def __init__(self, output_path, force = False):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.force = force
    
    def _file_exists(self, path: Path, force: bool) -> bool:
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
    
    def normalize_fever_evidence(self, record: dict) -> list[dict]:
        normalized = []
        for evidence_group in record.get("evidence", []):
            for item in evidence_group:
                if isinstance(item, list) and len(item) == 4:
                    normalized.append({
                        "annotation_id": item[0],
                        "evidence_id": item[1],
                        "wikipedia_title": item[2],
                        "sentence_id": item[3]
                    })
        return normalized

    
    def convert_json_to_jsonl(self, json_path: Path, jsonl_path: Optional[Path] = None, force: bool = False) -> Optional[Path]:
        json_path = Path(json_path)
        if jsonl_path is None: 
            jsonl_path = json_path.with_suffix('.jsonl')
        
        if jsonl_path.exists() and not force:
            tqdm.write(f"Skipping conversion, file already exists: {jsonl_path}")
            return jsonl_path
        
        try: 
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError(f"Expected a list of objects in {json_path}, got {type(data)}")

            with open(jsonl_path, 'w', encoding='utf-8') as f_out:
                for item in data:
                    json.dump(item, f_out)
                    f_out.write('\n')
            
            if "fever_dev_train" in json_path.name.lower():
                item["evidence"] = self.normalize_fever_evidence(item)
            
            tqdm.write(f"Converted {json_path} to {jsonl_path}")
            return jsonl_path

        except Exception as e:
            tqdm.write(f"Failed to convert {json_path} to JSONL: {e}")
            return None
    
    
    def convert_jsonl_to_csv(self, jsonl_path: str, csv_name: str):
        csv_path = self.output_path / f"{csv_name}.csv"
        
        if self._file_exists(csv_path, self.force):
            return csv_path
        
        try:
            df = pd.read_json(jsonl_path, lines=True)
            df.to_csv(csv_path, index=False)
            tqdm.write(f"Converted {jsonl_path} to {csv_path}")
            return csv_path
        except Exception as e:
            tqdm.write(f"Failed to convert {jsonl_path} to CSV: {e}")
            return None
        