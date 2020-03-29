import os

import spacy
from spacy.pipeline import EntityRuler

output_dir = os.getcwd()
modelo_spacy = "modelo"
print('curr {}'.format(modelo_spacy))

if os.name != 'nt':
    spacy.prefer_gpu()
nlp = spacy.load(modelo_spacy)  # Loads the spacy en model into a python object

ruler = EntityRuler(nlp)
patterns = [
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "mensuales"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "pesos"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "semanales"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "semanal"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "anuales"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "anual"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "quincenal"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "mensual"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "diario"}]},
    {"label": "SALARIO", "pattern": [{"LIKE_NUM": True}, {"LOWER": "quincenal"}]},

    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "mensuales"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "pesos"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "semanales"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "semanal"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "anuales"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "anual"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "quincenal"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "mensual"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "diario"}]},
    {"label": "SALARIO", "pattern": [{"IS_ASCII": True}, {"LOWER": "quincenal"}]},

]

rangos_salarios = [{"semana": 5.5}, {"quincena": 11.91}, {"mensual": 23.83}]
tipo_salarios = ["mensual", "semanal", "quincenal", "diaro", "anual"]

ruler.add_patterns(patterns)
nlp.add_pipe(ruler)


def listado_palabras(doc):
    """Buscar intenciones de un Doc y un listado de palabras de un texto"""

    palabras = {}
    for token in doc:
        palabras[token.text] = token.i
    return palabras


def buscar_intenciones(doc, palabras):
    """
           Funcion para buscar entidades e intenciones

           Parameters:
              doc (): Documento de spacy.
              palabras ([string]): Listado de palabras reconociddas por spacy.
           """
    tokens = []
    lista_intenciones = []
    for token in doc.ents:

        if token.text in palabras:
            ancestros = doc[palabras.get(token.text)]
            obj = {'intenciones': list(map(str, ancestros.ancestors))}
            tokens.append(obj)  # prints the text and POS

    for dic in tokens:
        for key in dic:
            for intencion in dic[key]:
                # print(intencion)
                if intencion not in lista_intenciones:
                    lista_intenciones.append(intencion)
        # for intencion in intenciones_.get("intenciones"):
        #     for inten in intencion:
        #

    return lista_intenciones


def entidades(doc):
    tokens = []
    for token in doc.ents:
        obj = token.label_
        if obj not in tokens:
            tokens.append(obj)
    tokens.sort()
    return tokens


def separar_oracion_tokens(oracion: str):
    return [token.text for token in nlp(oracion)]


def buscar_salario(oracion: str) -> int:
    """
    Funcion para buscar el salario con spacy

    :return: salario sino se encuentra -1.
    """

    salario = -1
    doc = nlp(oracion)
    tipo = list(
        set([lema.lemma_ for lema in doc]) & set(tipo_salarios)
    )
    print('tipo salario: {}'.format(tipo))
    for token in doc.ents:
        if token.label_ == "SALARIO":
            s = ''.join(x for x in token.text if x.isdigit())
            print('texto ')
            salario = int(s)

    if salario != -1:
        for rango in rangos_salarios:
            for key in rango:
                if tipo[0] is key:
                    valor_fijo_salario = rango[key]
                    salario = salario / valor_fijo_salario

    return salario
