import numpy as np
import xml.etree.ElementTree as ET
from itertools import chain
from collections import Counter
from functools import reduce
from tfidf import DocProcessor
from math import log
from tqdm import tqdm
import csv
cimport numpy as np

k3 = 100
alpha = 1.0
beta = 0.1
n = 5

def BM25(first, second, qtf):
    return first*second*(k3+1)*qtf/(k3+qtf)

class QueryProcessor():
    def __init__(self):
        self.vocab = {}
        self.docProcessor = DocProcessor()
        self.docProcessor.process()

        self.bigrams = self.docProcessor.get_bigram()
        #self.documents = self.docProcessor.get_documents()
        #self.vocabs = self.docProcessor.get_vocabs()
        self.first = self.docProcessor.get_first()
        self.second = self.docProcessor.get_second()
        self.voc = self.docProcessor.get_voc()
        self.tf = self.docProcessor.get_tf()
        self.doclens = self.docProcessor.get_doclens()
        self.avglen = self.docProcessor.get_avglen()
        self.N = 46972 
        self.results = []
        self.qids = []
        
        print("[*] start parsing queries...")
        with open("../model/vocab.all") as f:
            for i, line in enumerate(f):
                if i==0:
                    continue
                self.vocab[line.strip()] = i

        tree = ET.parse('../queries/query-train.xml')
        root = tree.getroot()
        self.queries = []
        self.qtf = [ [] for i in range(len(root))]
        self.qvoc = [ [] for i in range(len(root))]
        for child in root:
            qterms = child[4].text.strip().replace('。', '').split('、')
            qterms.append(child[1].text.strip().replace('。', ''))
            self.qids.append(child[0][-3:])
            self.queries.append(list(map(lambda x: list(map(lambda y: self.vocab[y], x)), qterms)))

        for i, query in enumerate(self.queries):
            self.queries[i] = list(map(lambda x: self.check_bigram(x), query))
            '''
            for terms in query:
                new_terms = self.check_bigram(terms)
                print(terms, ">>>>", new_terms)
            '''
        print("[*] finish parsing queries!")

    def check_bigram(self, terms):
        i=0
        while i<len(terms)-1:
            if str(terms[i])+' '+str(terms[i+1]) in self.bigrams.keys():
                terms[i] = self.bigrams[str(terms[i])+' '+str(terms[i+1])]
                terms.pop(i+1)
            i+=1
        return terms
    
    def BM25_all( self, np.ndarray[np.double_t, ndim=1] second, \
                        np.ndarray[np.int_t, ndim=1] voc, \
                        np.ndarray[np.double_t, ndim=1] qtf, \
                        np.ndarray[np.int_t, ndim=1] qvoc):

        cdef float score = 0
        cdef int p1=0, p2=0

        while p1<second.size and p2<qtf.size:
            if voc[p1] < qvoc[p2]:
                p1+=1
            elif voc[p1] > qvoc[p2]:
                p2+=1
            else:
                score += self.first[voc[p1]]*second[p1]*qtf[p2]
                p1+=1
                p2+=1
        return score
    
    def Rocchio(self, q, tf, voc):
        pool = []
        for i in range(n):
            for j in range(len(tf)):
                pool += [voc[i][j] for k in range(tf[i][j])]
        pairs = Counter(pool)

        for v, tf in pairs.items():
            if v in q.keys():
                q[v] = alpha*float(q[v])+beta*float(pairs[v])/n
            else:
                q[v] = beta*float(pairs[v])/n
        return q

    def process(self):
        print("[*] start processing query")
        progress = tqdm(total=len(self.queries))
        for qid, query in enumerate(self.queries): # query = [ [char1, char2, char3], [char4, char5] ]
            progress.update(1)
            qtfs = Counter(list((chain.from_iterable(query))))
            qvoc = np.array(list(qtfs.keys()))
            qtf = np.array(list(qtfs.values()))
            scores = []
            for i in range(self.N):
                if self.doclens[i]==0:
                    continue
                score = self.BM25_all(self.second[i], self.voc[i], qtf.astype(np.double), qvoc)
                if score>0:
                    scores.append([i, score])
            scores.sort(key=lambda x: x[1], reverse=True)
            #print("score: ", scores)
            candidate_tf = []
            candidate_voc = []
            for s in scores[:n]:
                candidate_tf.append(self.tf[s[0]])
                candidate_voc.append(self.voc[s[0]])
            qtfs = self.Rocchio(qtfs, candidate_tf, candidate_voc)
            
            qvoc = np.array(list(qtfs.keys()))
            qtf = np.array(list(qtfs.values()))
            scores = []
            for i in range(self.N):
                if self.doclens[i]==0:
                    continue
                score = self.BM25_all(self.second[i], self.voc[i], qtf, qvoc)
                if score>0:
                    scores.append([i, score])
            scores.sort(key=lambda x: x[1], reverse=True)
            self.results.append([])
            for s in scores[:100]:
                self.results[-1].append(s[0])

    def save(self):
        with open('submission.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['query_id', 'retrieved_docs'])
            for i in range(len(self.qids)):
                result = ""
                for d in self.results[i]:
                    result+= ("doc_"+str(d)+" ")
                result = result.strip()
                writer.writerow([str(self.qids[i]), result])

if __name__ == "__main__":
    QP = QueryProcessor()
    QP.process()
