from pathlib import Path


def get_project_dir() -> Path:
    return Path(__file__).resolve().parents[3]