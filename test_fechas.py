



from vocab import buscar_escala
from durations_nlp.duration import Duration
import dateparser
from datetime import datetime
import parsedatetime as pdt

c = pdt.Constants(localeID="es", usePyICU=False)
p = pdt.Calendar(c)


def buscar_dias(oracion: str):
    formato_fecha = buscar_escala(oracion)
    print('fecha 1 calculada {}'.format(formato_fecha))

    if formato_fecha is not None:
        duracion = Duration(formato_fecha)
        return duracion.to_days()

    time_struct, parse_status = p.parse(oracion)
    fecha2 = datetime(*time_struct[:6])
    print('fecha 2 calculada {}'.format(fecha2))

    if parse_status is 1:
        dias2 = calcular_dias(fecha2, datetime.utcnow())
        return dias2

    fecha = dateparser.parse(oracion, languages=["es"])
    print('fecha 3 calculada {}'.format(fecha))

    if fecha is not None:
        dias = calcular_dias(fecha, datetime.utcnow())
        return dias


    return None

def calcular_dias(d1, d2):
    return abs((d2 - d1).days)

oraciones = [
    "Tengo 3 semanas trabajando",
    "Estoy trabajando desde el 2 diciembre 2019",
    "tengo 3 meses trabajando",
    "Entre en febrero",
]

for oracion in oraciones:
    dias = buscar_dias(oracion)
    print("oracion: {} dias: {}".format(oracion, dias))

