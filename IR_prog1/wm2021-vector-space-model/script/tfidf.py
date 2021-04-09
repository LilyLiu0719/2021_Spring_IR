import numpy as np
from math import log
import time
from tqdm import tqdm
import pickle

k1 = 1.5
b = 0.75
k3 = 100
load = True

def BM25_first(N, df):
    return log( (N-df+0.5)/(df+0.5) )

def BM25_second(tf, dl, avdl):
	K = k1 * ( (1-b) + b * (float(dl)/float(avdl)) )
	return ( (k1 + 1) * tf ) / (K + tf)

class DocProcessor():
    def __init__(self):
        self.inverted_file = '../model/inverted-file'
        #self.inverted_file = '../model/inverted-file-test'
        #self.file_list = '../model/file-list'
        self.N = 46972

        self.inverted_dict = {}
        self.bigram_mapping = {}
        self.doc_vector = list(map(lambda x: [], range(self.N)))
        self.first = list(map(lambda x: [], range(29906)))

    def process(self):
        print("[*] start processing documents...")
        start_time = time.time()
        
        if load:
            with open('./obj/bigrams.pkl', 'rb') as f: 
                self.bigram_mapping = pickle.load(f)
            with open('./obj/docvec.pkl', 'rb') as f: 
                self.doc_vector = pickle.load(f)
            with open('./obj/first.pkl', 'rb') as f: 
                self.first = pickle.load(f)
            with open('./obj/inverted.pkl', 'rb') as f: 
                self.inverted_dict = pickle.load(f)
            print("[*] finish loading params from obj/")
        else:
            vocab_len = 29906 
            progress = tqdm(total=37320537)
            with open(self.inverted_file, 'r') as f:
                start = True
                for line in f:
                    progress.update(1)
                    assert(len(self.first)==vocab_len)
                    if start:
                        count = 0
                        start = False 
                        voc1, voc2, df = line.split()
                        df = int(df)
                        first = BM25_first(self.N, df)
                        if voc2 == "-1":
                            voc = int(voc1)
                            self.inverted_dict[voc] = []
                            self.first[voc] = first
                        else:
                            vocab_len+=1
                            voc = vocab_len
                            self.inverted_dict[voc] = []
                            key = voc1 + " " + voc2
                            self.bigram_mapping[key] = vocab_len
                            self.first.append(first)
                            
                    else:
                        doc_id, tf = line.split()
                        tf = int(tf)
                        count+=1
                        self.inverted_dict[voc].append([doc_id, tf])
                        self.doc_vector[int(doc_id)].append([voc, tf])
                        if count == df:
                            start = True
                self.save() 

            non_empty_vector = list(filter(lambda x: len(x)>0, self.doc_vector))
            non_empty_index = list(filter(lambda x: len(self.doc_vector[x])>0, range(len(self.doc_vector))))
            empty_index = list(filter(lambda x: len(self.doc_vector[x])==0, range(len(self.doc_vector))))
            print("index: ", empty_index)
            #print("doc_vector:", non_empty_vector)
            doclens = list(map(lambda y: y[1], list(map(lambda x: np.array(x).sum(axis=0), non_empty_vector))))
            for i in empty_index:
                doclens.insert(i, 0)
            print(len(doclens))
            avglen = sum(doclens)/len(doclens)
            #print("doclen =", doclens)
            for i in non_empty_index:
                print(i)
                self.doc_vector[i] = list(map(lambda x: [x[0], BM25_second(x[1], doclens[i], avglen)], self.doc_vector[i]))
            
            print("[*] successfully generated document VSM")
            print("total dimension:", vocab_len)
            print("time:", time.time()-start_time)

    def save(self):
        with open('./obj/bigrams.pkl', 'wb') as f: 
            pickle.dump(self.bigram_mapping, f)
        with open('./obj/docvec.pkl', 'wb') as f: 
            pickle.dump(self.doc_vector, f)
        with open('./obj/first.pkl', 'wb') as f: 
            pickle.dump(self.first, f)
        with open('./obj/inverted.pkl', 'wb') as f: 
            pickle.dump(self.inverted_dict, f)

    def get_bigram(self):
        return self.bigram_mapping
    
    def get_first(self):
        return self.first

    def get_docvec(self):
        return self.doc_vector

    def get_inverted(self):
        return self.inverted_dict

if __name__ == "__main__":
    tfidf = DocProcessor()
    tfidf.process()
    #tfidf.save()
