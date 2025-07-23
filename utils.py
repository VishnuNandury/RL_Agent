import json
from pathlib import Path

INTERACTIONS_FILE = Path("interactions.json")

def save_interaction(interaction):
    if INTERACTIONS_FILE.exists():
        data = json.loads(INTERACTIONS_FILE.read_text())
    else:
        data = []
    data.append(interaction)
    INTERACTIONS_FILE.write_text(json.dumps(data, indent=2))

def load_interactions():
    if INTERACTIONS_FILE.exists():
        return json.loads(INTERACTIONS_FILE.read_text())
    return []
