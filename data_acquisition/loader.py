from pathlib import Path
from .downloader import DataDownLoader
from tqdm import tqdm

def main(force: bool = False, prompt_user: bool = True):
    urls = {
        "hotpot_train": 
            {"url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json", 
            "type": "json"},
        "hotpot_dev_distractor": 
            {"url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "type": "json"},
        "hotpot_dev_fullwiki": 
            {"url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
            "type": "json"},
        "fever_dev_train": {"url": "https://fever.ai/download/fever/shared_task_dev.jsonl",
                            "type": "jsonl"},
        "truthful_qa_train": {"url": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
                            "type": "csv"}
    }

    hf_datasets = [
        {"name": "squad_v2", "split": "train", "format": "csv"},
        {"name": "nq_open", "split": "train", "format": "jsonl"}
    ]

    downloader = DataDownLoader(output_path=Path("../data/raw"))

    # Prompt user only if requested
    if prompt_user:
        force_download = input("Do you want to overwrite existing files? (yes/no): ").strip().lower() == 'yes'
        downloader.force = force_download
    else:
        downloader.force = force

    # Download URL datasets
    print("Downloading datasets from URLs...\n")
    for name in tqdm(urls, desc="URL Downloads"):
        info = urls[name]
        if info["type"] == "json":
            downloader.download_json(name, info["url"])    
        elif info["type"] == "jsonl":
            downloader.url_download(info["url"], f"{name}.jsonl")
        elif info["type"] == "csv":
            downloader.download_csv(name, info["url"])

    # --- Hugging Face Downloads ---
    print("\nDownloading datasets from Hugging Face...\n")
    for config in tqdm(hf_datasets, desc="Hugging Face Downloads"):
        downloader.hugging_face_download(
            dataset_name=config["name"],
            subset=config.get("subset"),
            split=config.get("split", "train"),
            file_format=config.get("format", "jsonl")
        )
    
    # Convert JSON to JSONL if needed
    raw_dir = Path("../data/raw")
    json_files = list(raw_dir.glob("*.json"))
    
    if json_files:
        tqdm.write("\nConverting JSON files to JSONL format...\n")
        for json_path in tqdm(json_files, desc="Converting JSON to JSONL"):
            tqdm.write(f"Converting {json_path.name} to JSONL format")
            jsonl_path = downloader.convert_json_to_jsonl(json_path)
            
            if jsonl_path:
                # Archive the original .json file
                archive_path = json_path.parent / "archive"
                archive_path.mkdir(exist_ok=True)
                json_path.rename(archive_path / json_path.name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and cache all datasets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if they already exist."
    )

    args = parser.parse_args()
    main(force=args.force)