import ast
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Optional
from data_acquisition.downloader import DataDownLoader
from tqdm import tqdm
import pandas as pd

class DataCleaner:
    def __init__(self, output_path: Path, input_path: Path) -> None:
        self.output_path = output_path
        self.input_path = input_path
        
    def convert_json_to_jsonl(self, json_path: Path, jsonl_path: Optional[Path] = None, force: bool = False) -> Optional[Path]:
        json_path = Path(json_path)
        if jsonl_path is None: 
            jsonl_path = json_path.with_suffix('.jsonl')
        
        if jsonl_path.exists() and not force:
            tqdm.write(f"Skipping conversion, file already exists: {jsonl_path}")
            return jsonl_path
        
        try: 
            with open(json_path, 'r', encoding='utf-8') as f:
                # Try to load as a list
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError(f"Expected list of records, got {type(data)}")
                except json.JSONDecodeError as e:
                    tqdm.write(f"JSON decode failed with: {e}. Assuming file is already JSONL.")
                    shutil.copy2(json_path, jsonl_path)
                    return jsonl_path

            with open(jsonl_path, 'w', encoding='utf-8') as f_out:
                for item in data:
                    json.dump(item, f_out)
                    f_out.write('\n')
            
            tqdm.write(f"Converted {json_path} to {jsonl_path}")
            return jsonl_path

        except Exception as e:
            tqdm.write(f"Failed to convert {json_path} to JSONL: {e}")
            return None
    
    
    def convert_jsonl_to_csv(self, jsonl_path: str, csv_name: str):
        csv_path = self.output_path / f"{csv_name}.csv"
        
        if DataDownLoader._file_exists(csv_path, force=False):
            tqdm.write(f"Skipping conversion, file already exists: {csv_path}")
            return csv_path
        if csv_path.exists():
            return csv_path
        try:
            df = pd.read_json(jsonl_path, lines=True)
            df.to_csv(csv_path, index=False)
            tqdm.write(f"Converted {jsonl_path} to {csv_path}")
            return csv_path
        except Exception as e:
            tqdm.write(f"Failed to convert {jsonl_path} to CSV: {e}")
            return None    
    
    
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
    
    def clean_hotpotqa_record(self, record: dict) -> dict:
        # Normalize supporting_facts
        if "supporting_facts" in record:
            record["supporting_facts"] = [
                {"title": fact[0], "sentence_id": fact[1]}
                for fact in record["supporting_facts"]
                if isinstance(fact, list) and len(fact) == 2
            ]

        # Normalize context into a flat list of sentence records
        if "context" in record:
            new_context = []
            for para in record["context"]:
                if isinstance(para, list) and len(para) == 2:
                    title, sentences = para
                elif isinstance(para, dict) and "title" in para and "sentences" in para:
                    title, sentences = para["title"], para["sentences"]
                else:
                    continue
                for i, sent in enumerate(sentences):
                    new_context.append({
                        "title": title,
                        "sentence_id": i,
                        "text": sent
                    })
            record["context"] = new_context

        return record
    
    def clean_squad_answers(self, csv_path: Path, force: bool = False):
        cleaned_path = self.output_path / csv_path.name
        failed_path = self.input_path / f"{csv_path.stem}_failed.csv"

        if cleaned_path.exists() and not force:
            tqdm.write(f"Skipping cleaning, file already exists: {cleaned_path}")
            return cleaned_path

        tqdm.write(f"Cleaning SQuAD CSV: {csv_path.name}")

        try:
            df = pd.read_csv(csv_path, quotechar='"', escapechar='\\')
            required_columns = {"id", "title", "context", "question", "answers"}
            valid_rows = []
            failed_rows = []

            answer_col = "answers" if "answers" in df.columns else "answer"

            for _, row in df.iterrows():
                # Check for missing or null required fields
                if not required_columns.issubset(row.index) or row[list(required_columns)].isnull().any():
                    row["parse_error"] = "Missing required columns or null value"
                    failed_rows.append(row.to_dict())
                    continue

                val = row.get(answer_col, "")
                try:
                    # Fix array(...) syntax
                    val = re.sub(r"array\((\[[^\]]*\])(?:,\s*dtype=[^)]+)?\)", r"\1", val)
                    parsed = ast.literal_eval(val)

                    text_vals = parsed.get("text", [])
                    start_vals = parsed.get("answer_start", [])

                    if not isinstance(text_vals, (list, tuple)):
                        text_vals = [str(text_vals)]
                    if not isinstance(start_vals, (list, tuple)):
                        start_vals = [int(start_vals)]

                    row[answer_col] = json.dumps({
                        "text": text_vals,
                        "answer_start": start_vals
                    })

                    # Normalize text fields
                    for field in ["context", "question", "title"]:
                        row[field] = str(row[field]).replace('"', '""').replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').strip()

                    valid_rows.append(row)

                except Exception as e:
                    row["parse_error"] = str(e)
                    failed_rows.append(row.to_dict())

            # Save cleaned data
            pd.DataFrame(valid_rows).to_csv(cleaned_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
            tqdm.write(f"Saved cleaned CSV to: {cleaned_path}")

            # Save failures to raw directory
            if failed_rows:
                pd.DataFrame(failed_rows).to_csv(failed_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
                tqdm.write(f"Saved {len(failed_rows)} failed rows to: {failed_path}")

            return cleaned_path

        except Exception as e:
            tqdm.write(f"Failed to clean {csv_path.name}: {e}")
            return None


    def get_cleaning_function(self, filename: str) -> Optional[str]:
        if "fever" in filename.lower():
            return "fever"
        elif "hotpot" in filename.lower():
            return "hotpot" 
        elif "nq_open" in filename.lower():
            return "nq_open"
        elif "squad" in filename.lower():
            return "squad"
        return None
    
    def clean_jsonl(self, jsonl_path: Path, dataset: str, force: bool = False):
        cleaned_path = self.output_path / jsonl_path.name
        if cleaned_path.exists() and not force:
            tqdm.write(f"Skipping cleaning, file already exists: {cleaned_path}")
            return cleaned_path
        
        
        with open(jsonl_path, 'r', encoding='utf-8') as f_in, open(cleaned_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc=f"Cleaning {jsonl_path.name}"):
                record = json.loads(line)
                if dataset == "fever":
                    record["evidence"] = self.normalize_fever_evidence(record)
                elif dataset == "hotpot":
                    record = self.clean_hotpotqa_record(record)
                json.dump(record, f_out)
                f_out.write('\n')
