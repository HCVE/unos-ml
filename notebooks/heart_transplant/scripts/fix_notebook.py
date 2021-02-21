import argparse

import json

parser = argparse.ArgumentParser()
parser.add_argument('--notebook')
args = parser.parse_args()

with open(args.notebook) as f:
    content = json.load(f)

for cell in content['cells']:
    if 'outputs' not in cell:
        cell['outputs'] = []

with open(args.notebook, 'w') as f:
    json.dump(content, f)
