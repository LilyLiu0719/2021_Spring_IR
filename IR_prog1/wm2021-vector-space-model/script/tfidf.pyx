import numpy as np
from math import log
import time
from tqdm import tqdm
import pickle
cimport numpy as np

k1 = 1.5
b = 0.75
k3 = 100

def BM25_first(N, df):
    return np.log( (N-df+0.5)/(df+0.5) )

def BM25_second(np.ndarray[np.int_t, ndim=1] tf, int dl, float avdl):
    cdef float K = k1 * ( (1-b) + b * dl/avdl )
    return ( (k1 + 1) * tf ) / (K + tf)

class DocProcessor():
    def __init__(self):
        self.inverted_file = '../model/inverted-file'
        #self.inverted_file = '../model/inverted-file-test'
        #self.file_list = '../model/file-list'
        self.N = 46972

        #self.vocabs = [] # [ voc1, voc2, ... ] voc: [ df, first, [doc1, tf], [doc2, tf], ... ]
        self.voc = []
        self.tf = []
        self.df = []
        self.second = []
        #self.documents = list(map(lambda x: [], range(self.N))) # [doc1, doc2, ...] doc: [[term1, tf, second], [term2, tf, second], ...]
        self.bigram_mapping = {} # "v1 v2": new_voc

    def process(self, load=False):
        print("[*] start processing documents...")
        start_time = time.time()
        
        if load:
            print("[ ] start loading params from obj/")
            with open('./obj/bigrams.pkl', 'rb') as f: 
                self.bigram_mapping = pickle.load(f)
            with open('./obj/documents.pkl', 'rb') as f: 
                self.documents = pickle.load(f)
            with open('./obj/vocabs.pkl', 'rb') as f: 
                self.vocabs = pickle.load(f)
            print("[*] finish loading params from obj/")

        else:
            vocab_len = 29906 
            #self.vocabs = list(map(lambda x: [], range(vocab_len+1)))
            self.df = [ 0 for i in range(vocab_len+1)]
            self.tf = [ [] for i in range(self.N)]
            self.voc = [ [] for i in range(self.N)]
            self.second = [ [] for i in range(self.N)]
            with open(self.inverted_file, 'r') as f:
                start = True
                lines = f.readlines()
                for line in tqdm(lines):
                    if start:
                        count = 0
                        start = False 
                        l = line.strip().split()
                        df = int(l[2])
                        if l[1] == "-1":
                            voc = int(l[0])
                            self.df[voc] = df
                            #self.vocabs[voc] = []
                        else:
                            vocab_len+=1
                            voc = vocab_len
                            #self.vocabs.append([])
                            self.df.append(df)
                            key = l[0] + " " + l[1]
                            self.bigram_mapping[key] = vocab_len
                            
                    else:
                        l = line.strip().split()
                        doc_id = int(l[0])
                        tf = int(l[1])
                        count+=1
                        #self.vocabs[voc].append([doc_id, tf])
                        #self.documents[doc_id].append((voc, tf))
                        self.voc[doc_id].append(voc)
                        self.tf[doc_id].append(tf)
                        if count == df:
                            start = True

            self.first = BM25_first(self.N, np.array(self.df))
            print("first:", self.first)
            
            for i in range(self.N):
                if len(self.tf[i]) == 0:
                    self.tf[i].append(1)
                if len(self.voc[i]) == 0:
                    self.voc[i].append(1)
                self.tf[i] = np.array(self.tf[i])
                self.voc[i] = np.array(self.voc[i])

            #non_empty_vector = list(filter(lambda x: len(x)>0, self.documents))
            #non_empty_index = list(filter(lambda x: len(self.documents[x])>0, range(len(self.documents))))
            #empty_index = list(filter(lambda x: len(self.documents[x])==0, range(len(self.documents))))
            #print("empty index: ", empty_index)
            #print("doc_vector:", non_empty_vector)
            
            self.doclens = np.array(list(map(lambda x: x.sum(), self.tf)), dtype=np.int16)
            #for i in empty_index:
            #    self.doclens.insert(i, 0)
            self.avglen = sum(self.doclens)/len(self.doclens)
            print("avglen =", self.avglen)
            for i in range(self.N):
                self.second[i] = BM25_second(self.tf[i], self.doclens[i], self.avglen)
        
        print("[*] successfully generated document Okapi BM25 model")
        #print("total dimension:", vocab_len)
        print("time:", time.time()-start_time)

    def save(self):
        with open('./obj/bigrams.pkl', 'wb') as f: 
            pickle.dump(self.bigram_mapping, f)
        with open('./obj/documents.pkl', 'wb') as f: 
            pickle.dump(self.documents, f)
        with open('./obj/vocabs.pkl', 'wb') as f: 
            pickle.dump(self.vocabs, f)

    def get_bigram(self):
        return self.bigram_mapping
    
    def get_voc(self):
        return self.voc

    def get_tf(self):
        return self.tf

    def get_first(self):
        return self.first
    
    def get_second(self):
        return self.second

    def get_doclens(self):
        return self.doclens
    
    def get_avglen(self):
        return self.avglen

if __name__ == "__main__":
    tfidf = DocProcessor()
    tfidf.process()
    #tfidf.save()
