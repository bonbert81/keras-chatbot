import calendar
import locale
import os
from datetime import datetime

import dateparser
import parsedatetime as pdt
from durations_nlp import Duration
from spacy.pipeline import EntityRuler

from vocab import buscar_escala

import spacy

output_dir = os.getcwd()
modelo_spacy = "{}\modelo".format(output_dir)
print('curr {}'.format(modelo_spacy))

if os.name != 'nt':
    spacy.prefer_gpu()
nlp = spacy.load(modelo_spacy)  # Loads the spacy en model into a python object

locale.setlocale(locale.LC_TIME, "es_ES")  #
new_date = datetime.now()

print(new_date.strftime('%B'))

MESES = [month for month in calendar.month_name[1:]]

print('meses: {}'.format(MESES))

fecha = dateparser.parse("Estoy trabajando desde diciembre 2018", languages=["es"])
print('fecha: {}'.format(fecha))

c = pdt.Constants(localeID="es", usePyICU=False)
p = pdt.Calendar(c)

time_struct, parse_status = p.parse("Estoy trabajando desde diciembre 2018")
print('fecha2 : {} status {}'.format(datetime(*time_struct[:6]), parse_status))

for time_str in ['1 segundo', '2 minutos', '3 horas', '5 semanas', '6 meses', '7 años']:
    formato_fecha = buscar_escala(time_str)
    if formato_fecha is not None:
        duration = Duration(formato_fecha)
        print('formato fecha {} duracion {}'.format(formato_fecha, duration.to_days()))

oraciones = [
    "Cuanto me corresponde por concepto de vacaciones, tengo 5 años trabajando.",

    "Quiero saber cuanto sera mi salario de navidad cobro 20000 mensual y entre en febrero",

    "Cuanto me coresponde de vacaciones gano 10,435 mensuales y tengo 12 meses trabajando",

    "Si gano 2,105 semanales y labore 8 meses con 12 días, cuanto me corresponde de vacaciones",
    "1.	Sandra ha trabajado continuamente por 02 años y gana RD$25,000 mensual.",
    "2.	Valeria  trabajó 09 meses y ganaba RD$15,000 quincenal."
]

ruler = EntityRuler(nlp)
patterns = [
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "mensuales"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "pesos"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "semanales"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "anuales"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "anual"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "quincenal"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "mensual"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "diario"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "quincenal"}]},

]

ruler.add_patterns(patterns)
nlp.add_pipe(ruler)
for texto in oraciones:
    doc = nlp(texto)
    for token in doc.ents:
        print('texto {} entidad {}'.format(token.text, token.label_))
