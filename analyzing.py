# installing packages:
# python3 -m pip install deeppavlov
# python3 -m deeppavlov install morpho_ru_syntagrus_pymorphy
# python3 -m deeppavlov install syntax_ru_syntagrus_bert
# python3 -m pip install conllu
# python3 -m pip install razdel


import time
from tqdm import tqdm
import re
import json
import os
from deeppavlov import build_model, configs
from razdel import sentenize, tokenize
from conllu import parse


# importing deeppavlov model
dp_model = build_model("ru_syntagrus_joint_parsing", download=True)


def analyze_text(text):
    parsed_sentences = []
    if len(text) > 65 and not re.match('https?://', text):
        sentences = [sent.text for sent in list(sentenize(text))]
        for sent in sentences:
            if 'Продолжение читайте в газете "Вестник района"' not in sent:
                tokens = [token.text for token in list(tokenize(sent))]
                if len(tokens) < 400:
                    parsed_sentences.append((sent, parse(dp_model([tokens])[0])[0]))
    return parsed_sentences

def write_to_conllu(parsed_texts, name):
    with open('out/' + name + '.conllu', 'a', encoding = 'utf-8') as f:
        for sent in parsed_texts:
            f.write('# text = ' + sent[0] + '\n')
            f.write(sent[1].serialize())


list_of_files = [file for file in os.listdir() if '.txt' in file]
encodings = ['utf-8', 'cp1251']

for file in list_of_files:
    print('Start processing file: ', file)
    for enc in encodings:
        try:
            with open(file, 'r', encoding=enc) as f:
                result = f.readlines()
        except UnicodeDecodeError:
            pass
    for line in tqdm(result):
        texts = analyze_text(line)
        if texts:
            write_to_conllu(texts, file)
    print('Finished: ', file)
    print('-'*100)