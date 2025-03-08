#!/usr/bin/env python3
import os
import re

for root, dirs, files in os.walk('/Users/mps/projects/AI-PROJECTS/TorchDevice'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()

                # Remove trailing whitespace
                content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

                # Remove spaces on blank lines
                content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)

                with open(filepath, 'w') as f:
                    f.write(content)

                print(f'Cleaned {filepath}')
            except Exception as e:
                print(f'Error processing {filepath}: {e}')