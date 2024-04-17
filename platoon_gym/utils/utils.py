from pathlib import Path


def get_project_root() -> str:
    """
    Returns:
        str: project root path
    """
    return str(Path(__file__).parent.parent.parent)
