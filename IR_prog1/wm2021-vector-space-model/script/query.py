import numpy as np
import xml.etree.ElementTree as ET
from tfidf import DocProcessor

k3 = 100

def BM25(first, second, qtf):
    return first*second*(k3+1)*qtf/(k3+qtf)

class QueryProcessor():
    def __init__(self):
        self.vocab = {}
        self.docProcessor = DocProcessor()
        self.docProcessor.process()
        self.bigrams = self.docProcessor.get_bigram()
        self.first = self.docProcessor.get_first()
        self.second = self.docProcessor.get_docvec()
        self.inverted = self.docProcessor.get_inverted()

        with open("../model/vocab.all") as f:
            for i, line in enumerate(f):
                if i==0:
                    continue
                self.vocab[line.strip()] = i
        tree = ET.parse('../queries/query-train.xml')
        root = tree.getroot()
        self.queries = []
        for child in root:
            qterms = child[4].text.strip().replace('。', '').split('、')
            print(qterms)
            self.queries.append(list(map(lambda x: list(map(lambda y: self.vocab[y], x)), qterms)))
        print("query:", self.queries)
        for i, query in enumerate(self.queries):
            #self.queries[i] = list(map(lambda x: self.check_bigram(x), self.queries[i]))
            for terms in query:
                new_terms = self.check_bigram(terms)
                print(terms, ">>>>", new_terms)
        print("query:", self.queries)

    def check_bigram(self, terms):
        i=0
        while i<len(terms)-1:
            if str(terms[i])+' '+str(terms[i+1]) in self.bigrams.keys():
                terms[i] = self.bigrams[str(terms[i])+' '+str(terms[i+1])]
                terms.pop(i+1)
            i+=1
        return terms
    
    def process(self):
        for query in self.queries:
            for doc, tf in self.inverted:
                score = BM25(self.first[query[]])




if __name__ == "__main__":
    QP = QueryProcessor()
