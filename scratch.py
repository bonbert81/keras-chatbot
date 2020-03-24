import calendar
import locale
from datetime import datetime

import dateparser
import parsedatetime as pdt
from durations_nlp import Duration

from vocab import buscar_escala

locale.setlocale(locale.LC_TIME, "es_ES")  #
new_date = datetime.now()

print(new_date.strftime('%B'))

MESES = [month for month in calendar.month_name[1:]]

print('meses: {}'.format(MESES))

fecha = dateparser.parse("marzo", languages=["es"])
print('fecha: {}'.format(fecha))

c = pdt.Constants(localeID="es", usePyICU=False)
p = pdt.Calendar(c)

time_struct, parse_status = p.parse("5 años")

print('fecha2 : {}'.format(datetime(*time_struct[:6])))

for time_str in ['1 segundo', '2 minutos', '3 horas', '5 semanas', '6 meses', '7 años']:
    formato_fecha = buscar_escala(time_str)
    if formato_fecha is not None:
        duration = Duration(formato_fecha)
        print('formato fecha {} duracion {}'.format(formato_fecha, duration.to_days()))

