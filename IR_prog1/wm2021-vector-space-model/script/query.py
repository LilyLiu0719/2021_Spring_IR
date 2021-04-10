import numpy as np
import xml.etree.ElementTree as ET
from itertools import chain
from collections import Counter
from functools import reduce
from tfidf import DocProcessor
from math import log
from tqdm import tqdm

k3 = 100
alpha = 1
beta = 0.1
n = 5

def BM25(first, second, qtf):
    return first*second*(k3+1)*qtf/(k3+qtf)

def pivote_nor(score, tf, dl, avdl, qtf, df):
    first = (1 + log(1+log(tf)) )/ ( (1-s)+s*dl/avdl )
    third = log( (N+1)/df )
    return first*qtf*third

def Rocchio(q, doc):
    pool = []
    for d in doc:
        for voc, tf in d:
            pool += [v for i in range(tf)]
    print("pool:", pool)
    tfs = Counter(pool)
    doc_keys = tfs.keys()
    for voc, qtf in q.items:
        if voc in tfs.keys():
            q[voc] += tfs[voc]/n
        else:
            q[voc] = tfs[voc]/n
    return q

class QueryProcessor():
    def __init__(self):
        self.vocab = {}
        self.docProcessor = DocProcessor()
        self.docProcessor.process()

        self.bigrams = self.docProcessor.get_bigram()
        self.documents = self.docProcessor.get_documents()
        self.vocabs = self.docProcessor.get_vocabs()
        self.doclens = self.docProcessor.get_doclens()
        self.avglen = self.docProcessor.get_avglen()

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
            self.queries.append(list(map(lambda x: list(map(lambda y: self.vocab[y], x)), qterms)))

        for i, query in enumerate(self.queries):
            self.queries[i] = list(map(lambda x: self.check_bigram(x), query))
            '''
            for terms in query:
                new_terms = self.check_bigram(terms)
                print(terms, ">>>>", new_terms)
            '''

    def check_bigram(self, terms):
        i=0
        while i<len(terms)-1:
            if str(terms[i])+' '+str(terms[i+1]) in self.bigrams.keys():
                terms[i] = self.bigrams[str(terms[i])+' '+str(terms[i+1])]
                terms.pop(i+1)
            i+=1
        return terms
    
    def process(self):
        print("[*] start processing query")
        progress = tqdm(total=len(self.queries))
        for query in self.queries: # query = [ [char1, char2, char3], [char4, char5] ]
            progress.update(1)
            qtfs = Counter(list((chain.from_iterable(query))))
            scores = []
            for i, doc in enumerate(self.documents):
                if len(doc)==0:
                    continue
                doc_dict = dict(np.array(doc)[:, :2])
                score = 0
                for voc, qtf in qtfs.items():
                    if voc in doc_dict.keys():
                        score += BM25(self.vocabs[voc][1], doc_dict[voc], qtf)
                #score = reduce(lambda x, y: BM25(self.vocabs[x[0]][1], doc_dict[][1], x[1]) + BM25(self.vocabs[y[0]][1], self.documents[doc][1], y[1]), qtfs.items() )
                if score>0:
                    scores.append([i, score])
            scores.sort(key=lambda x: x[1], reverse=True)
            print("score: ", scores)
            candidates = []
            for s in scores[:n]:
                candidates.append(self.documents[s[0]])
            new_query = Rocchio(query, candidates)
            print("new_query: ", new_query)
        
            qtfs = Counter(list((chain.from_iterable(new_query))))
            scores = []
            for i, doc in enumerate(self.documents):
                if len(doc)==0:
                    continue
                doc_dict = dict(np.array(doc)[:, :2])
                score = 0
                for voc, qtf in qtfs.items():
                    if voc in doc_dict.keys():
                        score += BM25(self.vocabs[voc][1], doc_dict[voc], qtf)
                #score = reduce(lambda x, y: BM25(self.vocabs[x[0]][1], doc_dict[][1], x[1]) + BM25(self.vocabs[y[0]][1], self.documents[doc][1], y[1]), qtfs.items() )
                if score>0:
                    scores.append([i, score])
            scores.sort(key=lambda x: x[1], reverse=True)
            print("new_score:", scores)

if __name__ == "__main__":
    QP = QueryProcessor()
    QP.process()
