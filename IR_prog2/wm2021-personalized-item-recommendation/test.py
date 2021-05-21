import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", '--out_path', type=str, default='results/submission.csv', help="save csv path")
parser.add_argument("-m", '--model_path', type=str, default='models/model.pth', help="model path")
parser.add_argument("-t", '--train_path', type=str, default='train.csv', help="train.csv path")
args = parser.parse_args()

n_factor = 512

lr = 1e-1
epochs = 500
batch_size = 2**8

val_ratio = 0.1

val_losses = []
train_losses = []
val_maps = []
train_maps = []

BCELoss = nn.BCELoss()
sigmoid = nn.Sigmoid()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("[*] running on", device)

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
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

def acc_fn(logit, label):
    logit = logit.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    pred = (logit>0).astype(np.int16)
    score = (pred==label).mean()
    return score

def MAP(ratings, uilist):
    ratings = ratings.cpu().detach().numpy()
    for i in range(len(uilist)):
        ratings[i][uilist[i]] = -10000
    ratings = np.argsort(-ratings, axis=1)
    return ratings[:, :50]

def sample_neg(pos, uilist_n):
    neg_sample = []
    for uid  in pos:
        negs = uilist_n[uid]
        niid = random.choice(negs)
        neg_sample.append(niid)
    return neg_sample

def main(model_path, out_path, train_path, n_user=4454, n_item=3260):
    df = pd.read_csv(train_path)
    users = df['UserId'].values
    items = df['ItemId'].values
    n_user = len(users)

    p_pairs = []
    
    uilist = [ [] for i in range(n_user) ]
    uilist_n = [ [] for i in range(n_user) ]

    max_iid = 0
    for uid in range(n_user):
        item_p = items[uid].split(' ')
        for i in item_p:
            iid = int(i)
            if iid > max_iid:
                max_iid = iid
            uilist[uid].append(iid)
            p_pairs.append((uid, iid))

    n_item = max_iid+1
    
    all_items = np.arange(n_item)
    for uid in range(n_user):
        items = np.array(uilist[uid])
        uilist_n[uid] = list(np.delete(all_items, items))
    model = torch.load(model_path)
    model.eval()

    users_tensor = torch.tensor(np.arange(n_user), dtype=torch.long).to(device)
    items_tensor = torch.tensor(np.arange(n_item), dtype=torch.long).to(device)

    ratings = model.rating(users_tensor, items_tensor)

    max50 = MAP(ratings, uilist)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['UserId', 'ItemId'])
        for i in range(n_user):
            tops = list(max50[i])
            items = ' '.join(str(v) for v in tops)
            writer.writerow([str(i), items])

if __name__ == "__main__":
    main(args.model_path, args.out_path, args.train_path)
