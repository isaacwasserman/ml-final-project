import os
import time

languages_0 = ["azeri","bengali","crimean-tatar"]
languages_1 = ["karelian","kashubian","maltese"]
languages_2 = ["middle-high-german","north-frisian","occitan"]
languages_3 = ["old-church-slavonic","pashto","tatar","livonian"]

languages = languages_3

print("start time:", time.time())

progress = 0
max_progress = len(languages * 3)
progress_bar = "[" + "".join([" "] * max_progress) + "]"

for language in languages:
    print(f'Working on {language}, regular')
    os.system(f'mv sigmorphon_data/{language}-hall-regular sigmorphon_data/{language}-hall')
    os.system(f'python inflection.py --datapath sigmorphon_data --L2 {language} --mode train --use_hall > train_logs/train-{language}-regular.log 2>&1')
    os.system(f'mv sigmorphon_data/{language}-hall sigmorphon_data/{language}-hall-regular')
    os.system(f'mv models/{language} models/{language}-hall-regular')
    progress += 1
    progress_bar = "[" + "".join(([u"\u2588"] * progress) + ([" "] * (max_progress - progress))) + "]"
    print(progress_bar)

    print(f'Working on {language}, ngram')
    os.system(f'mv sigmorphon_data/{language}-hall-ngram sigmorphon_data/{language}-hall')
    os.system(f'python inflection.py --datapath sigmorphon_data --L2 {language} --mode train --use_hall > train_logs/train-{language}-ngram.log  2>&1')
    os.system(f'mv sigmorphon_data/{language}-hall sigmorphon_data/{language}-hall-ngram')
    os.system(f'mv models/{language} models/{language}-hall-ngram')
    progress += 1
    progress_bar = "[" + "".join(([u"\u2588"] * progress) + ([" "] * (max_progress - progress))) + "]"
    print(progress_bar)

    print(f'Working on {language}, smart')
    os.system(f'mv sigmorphon_data/{language}-hall-smart sigmorphon_data/{language}-hall')
    os.system(f'python inflection.py --datapath sigmorphon_data --L2 {language} --mode train --use_hall > train_logs/train-{language}-smart.log  2>&1')
    os.system(f'mv sigmorphon_data/{language}-hall sigmorphon_data/{language}-hall-smart')
    os.system(f'mv models/{language} models/{language}-hall-smart')
    progress += 1
    progress_bar = "[" + "".join(([u"\u2588"] * progress) + ([" "] * (max_progress - progress))) + "]"
    print(progress_bar)