import pandas as pd
import numpy as np
import sys

output_path = sys.argv[1]
ans_path = "../queries/ans_train.csv"
df = pd.read_csv(output_path, dtype=str)
df_ans = pd.read_csv(ans_path, dtype=str)

ans_dict = {}
for index, row in df_ans.iterrows():
    ans_dict[row['query_id']] = row['retrieved_docs'].strip().split()

total_score = 0
for index, row in df.iterrows():
    query_score = 0
    count = 1
    for i, doc in enumerate(row['retrieved_docs'].strip().split()):
        if doc in ans_dict[row['query_id']]:
            query_score += count/(i+1)
            count += 1
    total_score += query_score / len(ans_dict[row['query_id']])
total_score /= len(ans_dict)
print(f"MAP score: {total_score}")
