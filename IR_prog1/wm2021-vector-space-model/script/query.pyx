import numpy as np
import xml.etree.ElementTree as ET
from itertools import chain
from collections import Counter
from functools import reduce
from tfidf import DocProcessor
from math import log
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
cimport numpy as np

alpha = 2.5
beta = 1
n = 7
k3 = 100

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
        self.file_name = {}

        print("bigram len:", len(self.bigrams))
        
        print("[*] start parsing queries...")
        with open("../model/vocab.all") as f:
            for i, line in enumerate(f):
                if i==0:
                    continue
                self.vocab[line.strip()] = i

        with open("../model/file-list") as f:
            for i, line in enumerate(f):
                name = line.strip().split('/')[-1]
                self.file_name[i] = name.lower()

        tree = ET.parse('../queries/query-test.xml')
        #tree = ET.parse('../queries/query-train.xml')
        root = tree.getroot()
        self.queries = []
        self.qtf = [ [] for i in range(len(root))]
        self.qvoc = [ [] for i in range(len(root))]
        for child in root:
            qterms = child[4].text.strip().replace('。', '').split('、')
            qterms.append(child[1].text.strip().replace('。', ''))
            self.qids.append(child[0].text[-3:])
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
        cdef np.ndarray[np.double_t, ndim=1] third = (k3+1)*qtf/(k3+qtf)

        while p1<second.size and p2<third.size:
            if voc[p1] < qvoc[p2]:
                p1+=1
            elif voc[p1] > qvoc[p2]:
                p2+=1
            else:
                score += self.first[voc[p1]]*second[p1]*third[p2]
                p1+=1
                p2+=1
        return score

    def Rocchio(self, q, tf, voc):
        pool = []
        for i in range(n):
            for j in range(len(tf[i])):
                pool += [voc[i][j]]*tf[i][j]
        pairs = Counter(pool)

        for v, tf in pairs.items():
            if v in q.keys():
                q[v] = alpha*float(q[v])+beta*float(pairs[v])/n
            else:
                q[v] = beta*float(pairs[v])/n
        return q
    '''
    def Rocchio(self, np.ndarray[np.int_t, ndim=1] qtf, np.ndarray[np.int_t, ndim=1] qvoc, list tf, list voc):
        cdef np.ndarray[np.double_t, ndim=1] new_qtf = np.array([], dtype=np.double)
        cdef np.ndarray[np.int_t, ndim=1] new_qvoc = np.array([], dtype=np.int)

        for i in range(1, n):
            new_qvoc, new_qtf = self.add_bow(new_qvoc, np.array(voc[i]), new_qtf, np.array(tf[i]).astype(np.double))
        new_qvoc, new_qtf = self.add_bow(new_qvoc, qvoc, beta*new_qtf, alpha*qtf.astype(np.double))
        return new_qvoc, new_qtf
    '''
    
    def add_bow(self, np.ndarray[np.int_t, ndim=1] voca, np.ndarray[np.int_t, ndim=1] vocb, \
                      np.ndarray[np.double_t, ndim=1] tfa, np.ndarray[np.double_t, ndim=1] tfb):

        cdef np.ndarray[np.int_t, ndim=1] all_voc = np.union1d(voca, vocb)
        cdef np.ndarray[np.double_t, ndim=1] all_tf = np.zeros(len(all_voc), dtype=np.double)
        all_tf[np.in1d(all_voc, voca)] += tfa
        all_tf[np.in1d(all_voc, vocb)] += tfb

        return all_voc, all_tf 

    def process(self):
        print("[*] start processing query")
        progress = tqdm(total=len(self.queries))
        for qid, query in enumerate(self.queries): # query = [ [char1, char2, char3], [char4, char5] ]
            progress.update(1)
            qtfs = dict(Counter(list((chain.from_iterable(query)))))
            qvoc = np.array(list(qtfs.keys()))
            qtf = np.array(list(qtfs.values()))
            qids = qvoc.argsort()
            qtf = qtf[qids]
            qvoc = qvoc[qids]
            scores = []
            for i in range(self.N):
                if self.doclens[i]==0:
                    continue
                score = self.BM25_all(self.second[i], self.voc[i], qtf.astype(np.double), qvoc)
                if score>0:
                    scores.append((i, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            #plt.bar(range(100), np.array(scores)[:100,1], alpha=0.5, label='old')
            candidate_tf = []
            candidate_voc = []
            for s in scores[:n]:
                candidate_tf.append(self.tf[s[0]])
                candidate_voc.append(self.voc[s[0]])
            #qvoc, qtf = self.Rocchio(qtf, qvoc, candidate_tf, candidate_voc)
            qtfs = self.Rocchio(qtfs, candidate_tf, candidate_voc)
            
            qvoc = np.array(list(qtfs.keys()))
            qtf = np.array(list(qtfs.values()))
            qids = qvoc.argsort()
            qtf = qtf[qids]
            qvoc = qvoc[qids]
            scores = []
            for i in range(self.N):
                if self.doclens[i]==0:
                    continue
                score = self.BM25_all(self.second[i], self.voc[i], qtf, qvoc)
                if score>0:
                    scores.append([i, score])

            scores.sort(key=lambda x: x[1], reverse=True)
            #score_arr = np.array(scores)[:, 1]
            #thres = np.median(score_arr) + np.std(score_arr)
            #print("thres: ", thres)
            self.results.append([])
            can_num = int(min(len(scores)*0.2, 100))
            plt.bar(range(can_num), np.array(scores)[:can_num,1])
            for s in scores[:can_num]:
                #print(s, end=' ')
                #if s[1] > thres:
                self.results[-1].append(s[0])
                #else:
                #    print("thres works!")
            plt.savefig('./vis/result'+str(qid)+'.jpg')
            plt.clf()

    def save(self):
        #with open('submission-train.csv', 'w', newline='') as csvfile:
        with open('submission-test.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['query_id', 'retrieved_docs'])
            for i in range(len(self.qids)):
                result = ""
                for d in self.results[i]:
                    if d in self.file_name.keys():
                        result+= (self.file_name[d]+" ")
                result = result.strip()
                writer.writerow([str(self.qids[i]), result])

if __name__ == "__main__":
    QP = QueryProcessor()
    QP.process()
