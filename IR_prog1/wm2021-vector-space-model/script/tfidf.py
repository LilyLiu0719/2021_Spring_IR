import numpy as np
from math import log
import time
from tqdm import tqdm
import pickle

k1 = 1.5
b = 0.75
k3 = 100

def BM25_first(N, df):
    return np.log( (N-df+0.5)/(df+0.5) )

def BM25_second(tf, dl, avdl):
	K = k1 * ( (1-b) + b * (float(dl)/float(avdl)) )
	return ( (k1 + 1) * tf ) / (K + tf)

class DocProcessor():
    def __init__(self):
        self.inverted_file = '../model/inverted-file'
        #self.inverted_file = '../model/inverted-file-test'
        #self.file_list = '../model/file-list'
        self.N = 46972

        self.vocabs = [] # [ voc1, voc2, ... ] voc: [ df, first, [doc1, tf], [doc2, tf], ... ]
        self.documents = list(map(lambda x: [], range(self.N))) # [doc1, doc2, ...] doc: [[term1, tf, second], [term2, tf, second], ...]
        self.df = []
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
            progress = tqdm(total=37320537)
            self.vocabs = list(map(lambda x: [], range(vocab_len+1)))
            self.df = list(map(lambda x: [], range(vocab_len+1)))
            with open(self.inverted_file, 'r') as f:
                start = True
                lines = f.readlines()
                for line in lines:
                    progress.update(1)
                    if start:
                        count = 0
                        start = False 
                        voc1, voc2, df = line.split()
                        df = int(df)
                        #first = BM25_first(self.N, df)
                        if voc2 == "-1":
                            voc = int(voc1)
                            self.df[voc] = df
                            self.vocabs[voc] = []
                        else:
                            vocab_len+=1
                            voc = vocab_len
                            self.vocabs.append([])
                            self.df.append(df)
                            key = voc1 + " " + voc2
                            self.bigram_mapping[key] = vocab_len
                            
                    else:
                        doc_id, tf = line.split()
                        tf = int(tf)
                        count+=1
                        self.vocabs[voc].append([doc_id, tf])
                        self.documents[int(doc_id)].append([voc, tf])
                        if count == df:
                            start = True

            first = BM25_first(self.N, np.array(self.df))
            print(first)
            non_empty_vector = list(filter(lambda x: len(x)>0, self.documents))
            non_empty_index = list(filter(lambda x: len(self.documents[x])>0, range(len(self.documents))))
            empty_index = list(filter(lambda x: len(self.documents[x])==0, range(len(self.documents))))
            print("empty index: ", empty_index)
            #print("doc_vector:", non_empty_vector)
            self.doclens = list(map(lambda y: y[1], list(map(lambda x: np.array(x).sum(axis=0), non_empty_vector))))
            for i in empty_index:
                self.doclens.insert(i, 0)
            self.avglen = sum(self.doclens)/len(self.doclens)
            #print("doclen =", doclens)
            for i in non_empty_index:
                self.documents[i] = list(map(lambda x: [x[0], x[1], BM25_second(x[1], self.doclens[i], self.avglen)], self.documents[i]))
        
        print("[*] successfully generated document VSM")
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
    
    def get_vocabs(self):
        return self.vocabs

    def get_documents(self):
        return self.documents

    def get_doclens(self):
        return self.doclens
    
    def get_avglen(self):
        return self.avglen

if __name__ == "__main__":
    tfidf = DocProcessor()
    tfidf.process()
    #tfidf.save()
