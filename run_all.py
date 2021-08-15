import subprocess

from include.utils import get_folder

commands = [
    # DEFAULT
    # Rolling CV
    [
        'default_classification.py', '--type', 'expanding', '--survival-days', '365', '--group',
        'ME_18'
    ],
    [
        'default_classification.py', '--type', 'expanding', '--survival-days', '365', '--group',
        'L_18'
    ],
    [
        'default_classification.py', '--type', 'expanding', '--survival-days', '90', '--group',
        'ME_18'
    ],
    # ['default_survival.py', '--type', 'expanding', '--survival-days', '365', '--group', 'ME_18'],
    # ['default_survival.py', '--type', 'expanding', '--survival-days', '365', '--group', 'L_18'],
    # ['default_survival.py', '--type', 'expanding', '--survival-days', '90', '--group', 'ME_18'],
    # 10-fold Shuffled CV
    [
        'default_classification.py', '--type', 'shuffled_cv', '--survival-days', '365', '--group',
        'ME_18'
    ],
    [
        'default_classification.py', '--type', 'shuffled_cv', '--survival-days', '365', '--group',
        'L_18'
    ],
    # ['default_survival.py', '--type', 'shuffled_cv', '--survival-days', '365', '--group', 'ME_18'],
    # ['default_survival.py', '--type', 'shuffled_cv', '--survival-days', '365', '--group', 'L_18'],
    # OPTIMIZED
    # Rolling CV
    [
        'optimized_classification.py', '--type', 'expanding', '--survival-days', '365', '--group',
        'ME_18'
    ],
    [
        'optimized_classification.py', '--type', 'expanding', '--survival-days', '365', '--group',
        'L_18'
    ],
    [
        'optimized_classification.py', '--type', 'expanding', '--survival-days', '90', '--group',
        'ME_18'
    ],
    # 10-fold Shuffled CV
    [
        'optimized_classification.py', '--type', 'shuffled_cv', '--survival-days', '365', '--group',
        'ME_18'
    ],
    [
        'optimized_classification.py', '--type', 'shuffled_cv', '--survival-days', '365', '--group',
        'L_18'
    ],
]

SCRIPT_FOLDER = get_folder(__file__)
for command in commands:
    with open(SCRIPT_FOLDER + '/logs/' + ' '.join(command), 'w') as log:
        subprocess.run(
            [
                'python',
                command[0],
                *command[1:],
            ],
            cwd=SCRIPT_FOLDER,
            # stdout=log,
        )
