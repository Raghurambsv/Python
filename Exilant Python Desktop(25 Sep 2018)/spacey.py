import spacy 
import os

from spacy.lang.en import English
nlp = spacy.load("en")

filename="//Users//raghuram.b//Desktop//chatbot//Rough worksheets//tripadvisor.txt"
#document = unicode(open(filename).read().decode('utf8')) 
document = nlp(document)


