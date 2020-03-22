import json
import os

import spacy

output_dir = os.getcwd()
modelo_spacy = "modelo"
print('curr {}'.format(modelo_spacy))

if os.name != 'nt':
    spacy.prefer_gpu()
nlp = spacy.load(modelo_spacy)  # Loads the spacy en model into a python object


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
