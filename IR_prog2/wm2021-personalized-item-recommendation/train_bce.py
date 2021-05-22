from comet_ml import Experiment
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import argparse
from utils import calc_ap

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("[*] running on", device)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("-e", "--epochs", type=int, default=10, help="epochs")
parser.add_argument("-b", '--batch_size', type=int, default=256, help="batch size")
parser.add_argument('--val_ratio', type=float, default=0.9, help="valid ratio")
parser.add_argument('-n', '--n_factor', type=int, default=128, help="hidden layer size")
parser.add_argument("-o", '--out_path', type=str, default='results/submission.csv', help="save csv path")
parser.add_argument("-m", '--model_path', type=str, default='models/model.pth', help="model path")
parser.add_argument("-t", '--train_path', type=str, default='train.csv', help="train.csv path")
args = parser.parse_args()

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)

    def forward(self, user, item):
        return (self.user_factors(user)*self.item_factors(item)).sum(1)

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

def top_k(ratings, uilist, k):
    ratings = ratings.cpu().detach().numpy()
    for uid in range(len(uilist)):
        ratings[uid][uilist[uid]] = -10000
    ratings = np.argsort(-ratings, axis=1)
    return ratings[:, :k]

def main(train_path):
    df = pd.read_csv(train_path)
    users = df['UserId'].values
    items = df['ItemId'].values
    n_user = len(users)

    p_pairs = []
    n_pairs = []
    
    uilist = [ [] for i in range(n_user) ]
    uilist_n = [ [] for i in range(n_user) ]
    train_uilist_n = [ [] for i in range(n_user) ]

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

    users_tensor = torch.tensor(users, dtype=torch.long).to(device)
    items_tensor = torch.tensor(np.arange(n_item), dtype=torch.long).to(device)
    
    all_items = np.arange(n_item)
    for uid in range(n_user):
        items = np.array(uilist[uid])
        negs = list(np.delete(all_items, items))
        negs = random.sample(negs, k=min( len(uilist[uid]), n_item-len(uilist[uid]) ) )
        for iid in negs:
            n_pairs.append((uid, iid))

    train_uilist_p = [ [] for i in range(n_user) ]
    val_uilist_p = [ [] for i in range(n_user) ]
    
    train_pairs_p = []
    val_pairs_p = []

    for uid in range(n_user):
        item_len = len(uilist[uid])
        val_num = int(item_len*args.val_ratio)
        random.shuffle(uilist[uid])
        train_uilist_p[uid] = uilist[uid][:val_num]
        val_uilist_p[uid] = uilist[uid][val_num:]
        for i in range(val_num):
            train_pairs_p.append((uid, train_uilist_p[uid][i]))
        for i in range(item_len-val_num):
            val_pairs_p.append((uid, val_uilist_p[uid][i]))
    
    for uid in range(n_user):
        items = np.array(train_uilist_p[uid])
        train_uilist_n[uid] = list(np.delete(all_items, items))
    p_pairs = np.array(p_pairs)
    n_pairs = np.array(n_pairs)
    
    np.random.shuffle(p_pairs)
    np.random.shuffle(n_pairs)
    
    val_n = int(len(p_pairs)*args.val_ratio)
    val_pairs_p = p_pairs[:val_n]
    val_pairs_n = n_pairs[:val_n]
    train_pairs_p = p_pairs[val_n:]
    train_pairs_n = n_pairs[val_n:]
    
    train_x = np.concatenate((train_pairs_p, train_pairs_n), axis=0)
    train_x = torch.tensor(train_x, dtype=torch.long)
    val_x = np.concatenate((val_pairs_p, val_pairs_n), axis=0)
    val_x = torch.tensor(val_x, dtype=torch.long)

    train_y = np.concatenate((np.ones(len(train_pairs_p)), np.zeros(len(train_pairs_n))), axis=0)
    train_y = torch.tensor(train_y, dtype=torch.float)
    val_y = np.concatenate((np.ones(len(val_pairs_p)), np.zeros(len(val_pairs_n))), axis=0)
    val_y = torch.tensor(val_y, dtype=torch.float)

    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=args.batch_size)

    model = MatrixFactorization(n_user, n_item, n_factors=args.n_factor).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    model.zero_grad()
    
    experiment = Experiment(api_key="vSQzzTkSbZxcZx3EZUPxdFLAr", project_name="IR_BCE")
    experiment.set_name("BCE")
    experiment.log_parameter("lr", args.lr)
    experiment.log_parameter("batch_size", args.batch_size)
    experiment.log_parameter("val ratio", args.val_ratio)
    experiment.log_parameter("n factor", args.n_factor)

    print("[*] start training")
    # train
    model.train()
    max_ap=0
    for e in range(1, args.epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        epoch_ap = 0
        for x_batch, y_batch in tqdm(train_dataloader):
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            users = x_batch[:, 0].to(device)
            items = x_batch[:, 1].to(device)
            y_pred = model(users, items)

            loss = loss_fn(y_pred, y_batch)
            acc = acc_fn(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc

        ratings = model.rating(users_tensor, items_tensor)
        max50 = top_k(ratings, [ [] for i in range(n_user) ], k=50)
        for uid in range(n_user):
            epoch_ap += calc_ap(max50[uid], train_uilist_p[uid])

        # eval
        model.eval()
        epoch_ap_val = 0
        max50 = top_k(ratings, train_uilist_p, k=50)
        for uid in range(n_user):
            epoch_ap_val += calc_ap(max50[uid], val_uilist_p[uid])

        experiment.log_metric("train_loss", epoch_loss/(args.batch_size*len(train_dataloader)), epoch=e)
        experiment.log_metric("train_ap", epoch_ap/n_user, epoch=e)
        experiment.log_metric("val_ap", epoch_ap_val/n_user, epoch=e)
        
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/(args.batch_size*len(train_dataloader)):.5f} | Map: {epoch_ap/n_user:.5f} | VAL Map: {epoch_ap_val/n_user:.5f}')

        if epoch_ap_val > max_ap:
            print("[*] model saved!")
            max_ap = epoch_ap_val
            torch.save(model, args.model_path)
            max50 = top_k(ratings, uilist, k=50)
            with open(args.out_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['UserId', 'ItemId'])
                for i in range(n_user):
                    tops = list(max50[i])
                    items = ' '.join(str(v) for v in tops)
                    writer.writerow([str(i), items])

if __name__ == "__main__":
    main('train.csv')
