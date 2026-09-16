"""User-editable find/replace dictionary for correcting transcript text."""
import json
from pathlib import Path

DEFAULT_PATH = Path("dictionary.json")


def load_dictionary(path=DEFAULT_PATH):
    """Load the replacement dictionary as {original: replacement}.

    Returns an empty dict if the file doesn't exist yet.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_dictionary(mapping, path=DEFAULT_PATH):
    """Write the replacement dictionary to disk as UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def apply_replacements(text, mapping):
    """Replace every occurrence of each key in `mapping` with its value."""
    for original, replacement in mapping.items():
        text = text.replace(original, replacement)
    return text
