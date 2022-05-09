import os

languages = ["azeri","bengali","crimean-tatar","ingrian","karelian","kashubian","livonian","maltese","middle-high-german","north-frisian","occitan","old-church-slavonic","pashto","tatar"]

for language in languages:
    print(f'Working on {language}, regular')
    os.system(f'python augment_regular.py sigmorphon_data {language} --examples 10000')
    print(f'Working on {language}, smart')
    os.system(f'python augment_smart.py sigmorphon_data {language} 1500 --examples 10000')
    print(f'Working on {language}, ngram')
    os.system(f'python augment_ngram.py sigmorphon_data {language} --examples 10000')