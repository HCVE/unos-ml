import argparse
import os
import subprocess
import sys
import tracemalloc

parser = argparse.ArgumentParser()
parser.add_argument('--cwd')
parser.add_argument('--script')
parser.add_argument('--script-args')
args = parser.parse_args()

script_args_parsed = args.script_args.split()

subprocess.run(
    [
        'python',
        args.script,
        *script_args_parsed,
    ],
    cwd=args.cwd,
)
