import random
import re
from datetime import datetime

import dateparser
import parsedatetime as pdt
from durations_nlp.duration import Duration

from laborabot import decode_sequence, str_to_tokens
from texto import buscar_intenciones, entidades, listado_palabras, nlp, buscar_salario
from vocab import DESPEDIDAS, DESPEDIDAS_BOT, SALUDOS, SALUDOS_BOT, buscar_escala

c = pdt.Constants(localeID="es", usePyICU=False)
p = pdt.Calendar(c)

calculo_vacaciones = [
    {"VACACIONES": ["corresponde", "toca"]},
    {"SALARIO": ["quiero", "toca", "corresponde"]},
]


def intent(texto: str):
    tokens = []
    data = []
    ents = []
    mapa = {}
    saludo = re.search("|".join(SALUDOS), texto)
    despedida = re.search("|".join(DESPEDIDAS), texto)
    # if saludo:
    #     obj = {"SALUDO": re.findall("|".join(SALUDOS), texto)}
    #     print("saludo ", obj)
    #
    #     resp = [
    #         w.replace("<obj>", re.findall("|".join(SALUDOS), texto).pop())
    #         for w in SALUDOS_BOT
    #     ]
    #
    #     tokens.append(resp[0])
    #     mapa = {"saludo": resp[0]}
    # elif despedida:
    #     obj = {"DESPEDIDA": re.findall("|".join(DESPEDIDAS), texto)}
    #     print("despedida ", obj)
    #     tokens.append(random.choice(DESPEDIDAS_BOT))
    #     mapa = {"despedida": random.choice(DESPEDIDAS_BOT)}
    #
    # else:
    doc = nlp(texto)  # Creates a doc object

    palabras = listado_palabras(doc)

    tokens = buscar_intenciones(doc, palabras)
    ent = entidades(doc)
    mapa = {"entidades": ent, "intenciones": tokens}

    return mapa


def visualizar(texto: str):
    doc = nlp(texto)  # Creates a doc object
    return doc


def calcular_vacaciones(oracion: str):
    """
    Función para calcular las vacaciones
    :param oracion: Texto original
    :return: True para realizar calculos y no predecir una respuesta, dias si existe, salario si existe
    """
    intencion_oracion = intent(oracion)
    calcular = False
    meses = -1
    salario = -1

    for item in calculo_vacaciones:
        for key in item:
            if key in intencion_oracion["entidades"]:
                intenciones = list(
                    set(intencion_oracion["intenciones"]) & set(item[key])
                )
                if intenciones:
                    meses = buscar_meses(oracion)
                    salario = buscar_salario(oracion)
                    print(
                        "key: {} intenciones: {}, intenciones encontradas {}, dias? {} salario? {}".format(
                            key, item[key], intenciones, meses, salario
                        )
                    )
                    calcular = True
                else:
                    print("no hay intenciones :(")

    texto = ""
    if salario != -1 and meses != -1:
        if meses < 5:
            texto = "Tiene menos de 5 meses trabajando, no le corresponde vacaciones."
        elif meses <= 11:
            vacaciones = round(meses * salario, 2)
            texto = "Le corresponde {} días de disfrute con {} de pago".format(meses + 1, vacaciones)
        elif meses < 60:
            vacaciones = round(14 * salario, 2)
            texto = "Le corresponde {} días de disfrute con {} de pago".format(14, vacaciones)
        else:
            vacaciones = round(18 * salario, 2)
            texto = "Le corresponde {} días de disfrute con {} de pago y 4 de pago".format(14, vacaciones)

    return calcular, texto


def predecir(oracion: str):
    input_seq = str_to_tokens(oracion)
    decoded_sentence = decode_sequence(input_seq).strip()
    decoded_sentence = decoded_sentence.replace(" end", ".")

    decoded_sentence = decoded_sentence.capitalize()
    print("res cap: {}".format(decoded_sentence))

    return decoded_sentence


def buscar_meses(oracion: str) -> int:
    """
    Funcion para calcular los meses de un string

    :return: meses si existe, sino -1
    """
    formato_fecha = buscar_escala(oracion)
    print("fecha 1 calculada {}".format(formato_fecha))

    if formato_fecha is not None:
        duracion = Duration(formato_fecha)
        return int(duracion.to_months())

    time_struct, parse_status = p.parse(oracion)
    fecha2 = datetime(*time_struct[:6])
    print("fecha 2 calculada {}".format(fecha2))

    if parse_status is 1:
        dias2 = calcular_meses(fecha2, datetime.utcnow())
        return dias2

    fecha = dateparser.parse(oracion, languages=["es"])
    print("fecha 3 calculada {}".format(fecha))

    if fecha is not None:
        dias = calcular_meses(fecha, datetime.utcnow())
        return dias

    return -1


def calcular_meses(d1, d2):
    return abs((d2 - d1).month)

# for seq_index in range(100):
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#     q = input("Enter question : ")
#     input_seq = str_to_tokens(q)
#     decoded_sentence = decode_sequence(input_seq)
#     print('-' * 5)
#     print('Input sentence:', q)
#     print('Decoded sentence:', decoded_sentence)
