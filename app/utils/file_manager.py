from pathlib import Path
import shutil

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_upload_file(upload_file, destination: Path) -> None:
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)