from pathlib import Path
from data_acquisition.cleaner import DataCleaner
from tqdm import tqdm
import shutil

def main():
    raw_path = Path("../data/raw")
    clean_path = Path("../data/clean")
    clean_path.mkdir(parents=True, exist_ok=True)
    
    cleaner = DataCleaner(output_path=clean_path, input_path=raw_path)
    
    for file_path in tqdm(raw_path.glob("*.*"), desc="Cleaning datasets"):
        if file_path.suffix == '.json':
            tqdm.write(f"Converting {file_path.name} to JSONL format")
            jsonl_path = cleaner.convert_json_to_jsonl(file_path, force=True)

            if jsonl_path:

                cleaner_name = cleaner.get_cleaning_function(jsonl_path.name)
                if cleaner_name:
                    cleaner.clean_jsonl(jsonl_path, cleaner_name, force=True)

                shutil.copy2(jsonl_path, clean_path / jsonl_path.name)
            
        elif file_path.suffix == '.jsonl':
            tqdm.write(f"Processing JSONL file: {file_path.name}")
            cleaner_name = cleaner.get_cleaning_function(file_path.name)
            if cleaner_name:
                # Always re-clean and overwrite the file in clean/
                cleaner.clean_jsonl(file_path, cleaner_name, force=True)
            else:
                # If no cleaning needed, copy raw file to clean/ anyway
                shutil.copy2(file_path, clean_path / file_path.name)
        
        elif file_path.suffix == '.csv':
            tqdm.write(f"Processing CSV file: {file_path.name}")
            cleaner_name = cleaner.get_cleaning_function(file_path.name)
            if cleaner_name:
                cleaner.clean_squad_answers(file_path, force=True)
            else:
                # If no cleaning needed, copy raw file to clean/ anyway
                tqdm.write(f"No specific cleaning function for {file_path.name}, copying as is.")
                shutil.copy2(file_path, clean_path / file_path.name)
    
    tqdm.write("Data cleaning completed successfully.")
    
if __name__ == "__main__":
    main()
    
    