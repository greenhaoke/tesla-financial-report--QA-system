import os
from pathlib import Path

def print_tree(dir_path: Path, prefix="", is_top=True):
    items = sorted(os.listdir(dir_path))
    # Filter out hidden or unnecessary files
    items = [i for i in items if not i.startswith('.') and i not in ['__pycache__', '1.0.0', '1.30.0']]
    
    dirs = [i for i in items if (dir_path / i).is_dir()]
    files = [i for i in items if (dir_path / i).is_file()]
    
    if not is_top:
        files = [] # Only show files at top level
        
    entries = dirs + files
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry}{'/' if entry in dirs else ''}")
        
        if entry in dirs:
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(dir_path / entry, next_prefix, is_top=False)

print("tesla/")
print_tree(Path("."))
