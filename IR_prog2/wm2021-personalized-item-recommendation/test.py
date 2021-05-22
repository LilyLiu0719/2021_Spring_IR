import torch
from torch import optim, nn
import csv
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", '--out_path', type=str, default='results/submission.csv', help="save csv path")
parser.add_argument("-m", '--model_path', type=str, default='models/model.pth', help="model path")
parser.add_argument("-t", '--train_path', type=str, default='train.csv', help="train.csv path")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("[*] running on", device)

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=256):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)

    def forward(self, user, item_p, item_n):
        user_embedding = self.user_factors(user)
        item_p_embedding = self.item_factors(item_p)
        item_n_embedding = self.item_factors(item_n)
        return (user_embedding*item_p_embedding).sum(1)-(user_embedding*item_n_embedding).sum(1)

    def rating(self, user, item):
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item).transpose(0, 1)
        ratings = torch.mm(user_embedding, item_embedding)
        return ratings

def top50(ratings, uilist):
    ratings = ratings.cpu().detach().numpy()
    for i in range(len(uilist)):
        ratings[i][uilist[i]] = -10000
    ratings = np.argsort(-ratings, axis=1)
    return ratings[:, :50]

def main(model_path, out_path, train_path, n_user=4454, n_item=3260):
    
    df = pd.read_csv(train_path)
    users = df['UserId'].values
    items = df['ItemId'].values
    n_user = len(users)
    
    uilist = [ [] for i in range(n_user) ]

    max_iid = 0
    for uid in range(n_user):
        item_p = items[uid].split(' ')
        for i in item_p:
            iid = int(i)
            if iid > max_iid:
                max_iid = iid
            uilist[uid].append(iid)

    n_item = max_iid+1
    
    if device == 'cpu':
        model = torch.load(model_path, map_location='cpu')
    else:
        model = torch.load(model_path)
    
    model.eval()

    users_tensor = torch.tensor(np.arange(n_user), dtype=torch.long).to(device)
    items_tensor = torch.tensor(np.arange(n_item), dtype=torch.long).to(device)

    ratings = model.rating(users_tensor, items_tensor)

    max50 = top50(ratings, uilist)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['UserId', 'ItemId'])
        for i in range(n_user):
            tops = list(max50[i])
            items = ' '.join(str(v) for v in tops)
            writer.writerow([str(i), items])

if __name__ == "__main__":
    main(args.model_path, args.out_path, args.train_path)
