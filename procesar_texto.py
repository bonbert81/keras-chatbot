import spacy

# Loads the spacy en model into a python object
nlp = spacy.load('es_core_news_sm')


def separar_oracion_tokens(oracion: str):
    return [token.text for token in nlp(oracion)]


print('tokens? {}'.format(separar_oracion_tokens('Qué es el contrato de trabajo?')))
