import re
import os

path = 'database/crud.py'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Cleanup previously broken lines
    stripped = line.strip()
    if stripped.startswith('with self.SessionFactory'):
        new_lines.append('        with self.SessionFactory() as session:\n')
    elif stripped.startswith('try:'):
        new_lines.append('            try:\n')
    elif stripped.startswith('except '):
        new_lines.append('            ' + stripped + '\n')
    elif stripped.startswith('session.') or stripped.startswith('return ') or stripped.startswith('gen =') or stripped.startswith('genome ='):
        # This is a bit risky but we'll try to indent it under try or with
        new_lines.append('                ' + stripped + '\n')
    else:
        new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
