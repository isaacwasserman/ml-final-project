import os
import time

languages_0 = ["azeri","bengali","crimean-tatar"]
languages_1 = ["karelian","kashubian","maltese"]
languages_2 = ["middle-high-german","north-frisian","occitan"]
languages_3 = ["old-church-slavonic","pashto","tatar","livonian"]

languages = languages_0 + languages_1 + languages_2 + languages_3

print("start time:", time.time())

progress = 0
max_progress = len(languages)
progress_bar = "[" + "".join([" "] * max_progress) + "]"

for language in languages:
    print(f'Working on {language}, unaugmented')
    os.system(f'python inflection.py --datapath sigmorphon_data --L2 {language} --mode train > train-{language}-no-augmentation.log 2>&1')
    os.system(f'mv models/{language} models/{language}-no-augmentation')
    progress += 1
    progress_bar = "[" + "".join(([u"\u2588"] * progress) + ([" "] * (max_progress - progress))) + "]"
    print(progress_bar)