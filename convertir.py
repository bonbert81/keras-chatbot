import yaml
import os
import csv

dir_path = "data"
files_list = os.listdir(dir_path + os.sep)


def grouplen(sequence, chunk_size):
    return  [sequence[i * chunk_size:(i + 1) * chunk_size] for i in range((len(sequence) + chunk_size - 1) // chunk_size )]  



for filepath in files_list:
    print("archivo: {}".format(dir_path + os.sep + filepath))
    if filepath.endswith(".csv"):
        stream = open(dir_path + os.sep + filepath, "r", encoding="utf-8")
        reader = csv.reader(stream, delimiter=",")

        preguntas = []
        for i, line in enumerate(reader):
            preguntas.append(line[1].strip())
            preguntas.append(line[2].strip())
        conv = ["conversaciones", grouplen(preguntas, 2)]
        with open("data/preguntas.yaml", "w", encoding="utf-8") as file:
            yaml.dump(grouplen(conv, 2), file)
