from query import QueryProcessor 
import time
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--rocchio', action="store_true", default=False)
parser.add_argument('--query', type=str, required=True)
parser.add_argument('--ranked', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--doc', type=str, required=True)

args = parser.parse_args()

start = time.time()
qp = QueryProcessor(args.model, args.doc, args.query, args.ranked, rocchio=args.rocchio)
qp.process()
qp.save()
print("[*] total time:", time.time()-start)
