import random
import re

from laborabot import str_to_tokens, decode_sequence
from texto import listado_palabras, buscar_intenciones, entidades, nlp
from vocab import DESPEDIDAS
from vocab import DESPEDIDAS_BOT
from vocab import SALUDOS
from vocab import SALUDOS_BOT


def intent(texto: str):
    tokens = []
    data = []
    ents = []
    mapa = {}
    saludo = re.search("|".join(SALUDOS), texto)
    despedida = re.search("|".join(DESPEDIDAS), texto)
    if saludo:
        obj = {"SALUDO": re.findall("|".join(SALUDOS), texto)}
        print("saludo ", obj)

        resp = [w.replace('<obj>', re.findall("|".join(SALUDOS), texto).pop()) for w in SALUDOS_BOT]

        tokens.append(resp[0])
        mapa = {'saludo': resp[0]}
    elif despedida:
        obj = {"DESPEDIDA": re.findall("|".join(DESPEDIDAS), texto)}
        print("despedida ", obj)
        tokens.append(random.choice(DESPEDIDAS_BOT))
        mapa = {'despedida': random.choice(DESPEDIDAS_BOT)}

    else:
        doc = nlp(texto)  # Creates a doc object

        palabras = listado_palabras(doc)

        tokens = buscar_intenciones(doc, palabras)
        ent = entidades(doc)
        mapa = {'entidades': ent, 'intenciones': tokens}

    return mapa


def visualizar(texto: str):
    doc = nlp(texto)  # Creates a doc object
    return doc


def predecir(oracion: str):
    input_seq = str_to_tokens(oracion)
    decoded_sentence = decode_sequence(input_seq).strip()
    decoded_sentence = decoded_sentence.replace(" end", ".")

    decoded_sentence = decoded_sentence.capitalize()
    print('res cap: {}'.format(decoded_sentence))

    return decoded_sentence

# for seq_index in range(100):
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#     q = input("Enter question : ")
#     input_seq = str_to_tokens(q)
#     decoded_sentence = decode_sequence(input_seq)
#     print('-' * 5)
#     print('Input sentence:', q)
#     print('Decoded sentence:', decoded_sentence)
