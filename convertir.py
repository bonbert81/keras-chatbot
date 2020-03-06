import yaml
import os
import csv

dir_path = 'data'
files_list = os.listdir(dir_path + os.sep)

for filepath in files_list:
    print('archivo: {}'.format(dir_path + os.sep + filepath))
    if filepath.endswith('.csv'):
        stream = open(dir_path + os.sep + filepath, 'r', encoding='utf-8')
        reader = csv.reader(stream, delimiter=",")

        preguntas = []
        for i, line in enumerate(reader):
            preguntas.append({line[1].strip(): [line[2]]})

        with open('data/preguntas.yaml', 'w', encoding='utf-8') as file:
            yaml.dump(preguntas, file)
