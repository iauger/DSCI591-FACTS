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
                            "type": "csv"},
        "squad_v2_train": {"url": "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/train-00000-of-00001.parquet",
                            "type": "parquet"},
        "squad_v2_validation": {"url": "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/validation-00000-of-00001.parquet",
                            "type": "parquet"}
        
    }

    hf_datasets = [
        {"name": "nq_open", "split": "train", "format": "jsonl"}
    ]

    downloader = DataDownLoader(output_path=Path("../data/raw"))

    # Prompt user only if requested
    if prompt_user:
        force_download = input("Do you want to overwrite existing files? (yes/no): ").strip().lower() == 'yes'
        downloader.force = force_download
    else:
        downloader.force = force

    raw_dir = Path("../data/raw")

    for name in tqdm(urls, desc="URL Downloads"):
        info = urls[name]
        stem = name.lower()

        already_exists = any(
            (raw_dir / f"{stem}{ext}").exists()
            for ext in [".json", ".jsonl", ".csv"]
        )

        if already_exists and not downloader.force:
            tqdm.write(f"Skipping {name}: cleaned or downloaded file already exists.")
            continue
        
        if info["type"] == "json":
            downloader.download_json(name, info["url"])    
        elif info["type"] == "jsonl":
            downloader.url_download(info["url"], f"{name}.jsonl")
        elif info["type"] == "csv":
            downloader.download_csv(name, info["url"])
        elif info["type"] == "parquet":
            downloader.download_and_load_parquet(info["url"], f"{name}.parquet", raw_dir, to_csv=True)

    # --- Hugging Face Downloads ---
    print("\nDownloading datasets from Hugging Face...\n")
    for config in tqdm(hf_datasets, desc="Hugging Face Downloads"):
        name = config["name"]
        split = config.get("split", "train")
        fmt = config.get("format", "jsonl")
        ext = ".csv" if fmt == "csv" else ".jsonl"
        filename_stem = f"{name}_{split}"
        file_exists = any((raw_dir / f"{filename_stem}{e}").exists() for e in [".csv", ".jsonl", ".json"])

        if file_exists and not downloader.force:
            tqdm.write(f"Skipping HuggingFace download for {filename_stem}: file already exists.")
            continue

        downloader.hugging_face_download(
            dataset_name=name,
            subset=config.get("subset"),
            split=split,
            file_format=fmt
        )

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