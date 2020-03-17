import spacy

nlp = spacy.load('es_core_news_sm')

oracion = 'Qué es el contrato de trabajo?'

  # Loads the spacy en model into a python object
print('tokens? {}'.format(nlp(oracion).tokens))