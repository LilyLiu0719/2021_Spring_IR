import numpy as np
from math import log
import time
from tqdm import tqdm

class TF_IDF():
    def __init__(self):
        self.inverted_file = '../model/inverted-file'
        #file_list = '../model/file-list'
        self.k = 1.5
        self.n = 46972

        #self.inverted_dict = {}
        self.bigram_mapping = {}
        self.doc_vector = list(map(lambda x: [], range(self.n)))

    def gen_tfidf(self):
        print("[*] start calculating TF-IDF...")
        start_time = time.time()
        vocab_len = 29906 
        progress = tqdm(total=37320537)
        with open(self.inverted_file, 'r') as f:
            start = True
            for line in f:
                progress.update(1)
                if start:
                    count = 0
                    start = False
                    voc1, voc2, _len = line.split()
                    _len = int(_len)
                    idf = log(self.n/_len)
                    if voc2 == "-1":
                        voc = int(voc1)
                        #self.inverted_dict[voc] = []
                    else:
                        vocab_len+=1
                        voc = vocab_len
                        #self.inverted_dict[voc] = []
                        key = voc1 + " " + voc2
                        self.bigram_mapping[key] = vocab_len
                        
                else:
                    doc_id, num = line.split()
                    num = int(num)
                    count+=1
                    #self.inverted_dict[voc].append([doc_id, num])
                    tf = ( (self.k+1)*num )/( num+self.k )
                    self.doc_vector[int(doc_id)].append([voc, tf*idf])
                    if count == _len:
                        start = True

            print("[*] successfully generated document VSM")
            print("total dimension:", vocab_len)
            print("time:", time.time()-start_time)
            #print("inverted_dict:", self.inverted_dict)
            #print("bigram_mapping:", self.bigram_mapping)
            #print("doc_vector:", self.doc_vector)

if __name__ == "__main__":
    tfidf = TF_IDF()
    tfidf.gen_tfidf()
