from query import QueryProcessor 
import time
start = time.time()
qp =  QueryProcessor()
qp.process()
qp.save()
print("[*] total time:", time.time()-start)
