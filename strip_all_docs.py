#!/usr/bin/env python3
import os
import re
from pathlib import Path

def strip_docstrings_and_comments(content):
    lines = content.split('\n')
    result = []
    in_multiline_string = False
    string_delimiter = None
    
    for line in lines:
        stripped_line = line.strip()
        
        if not in_multiline_string:
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                string_delimiter = '"""' if stripped_line.startswith('"""') else "'''"
                if stripped_line.count(string_delimiter) == 2:
                    continue
                else:
                    in_multiline_string = True
                    continue
        else:
            if string_delimiter in line:
                in_multiline_string = False
                string_delimiter = None
            continue
        
        if not in_multiline_string:
            comment_pos = line.find('#')
            if comment_pos != -1:
                in_string = False
                for i in range(comment_pos):
                    if line[i] in ['"', "'"]:
                        in_string = not in_string
                if not in_string:
                    line = line[:comment_pos].rstrip()
            
            if line.strip():
                result.append(line)
            elif line and not line.strip():
                if result and result[-1].strip():
                    result.append('')
    
    while result and not result[-1].strip():
        result.pop()
    
    return '\n'.join(result)

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned = strip_docstrings_and_comments(content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    python_files = []
    for root, dirs, files in os.walk('.'):
        if '.venv' in root or '__pycache__' in root or 'old_broken' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to clean")
    
    for filepath in python_files:
        if 'strip_all_docs.py' in filepath:
            continue
        print(f"Cleaning {filepath}...")
        process_file(filepath)
    
    print("Done!")

if __name__ == "__main__":
    main()