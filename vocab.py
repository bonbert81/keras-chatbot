import calendar
import locale
import re

locale.setlocale(locale.LC_TIME, "es_ES")

SALUDOS = ["Hola", "hola", "Hey", "hey", "Saludos", "Buenos día", "Buenos días", "Que tal"]

SALUDOS_BOT = ["<obj>, en que te puedo ayudar?"]

DESPEDIDAS = ["Adios", "Hasta luego", "Hasta la vista", "Bye", "bye"]
DESPEDIDAS_BOT = ["Adios", "Hasta luego", "Hasta la vista"]

MESES = [month for month in calendar.month_name[1:]]

escala_fecha = [{"month": ["mes", "meses"]}, {"day": ["dia", "día", "dias", "días"]}, {"year": ["año", "años"]},
                {"week": ["semana", "semanas", ]}
                ]

rangos_vacaciones = []


def buscar_escala(fecha: str):
    for dic in escala_fecha:
        for key in dic:
            if any(substring in fecha.split() for substring in dic[key]):
                numero = int(re.search(r'\d+', fecha).group())
                return "{} {}".format(numero, key)

    return None
