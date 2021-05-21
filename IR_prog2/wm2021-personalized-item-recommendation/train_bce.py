import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

n_factor = 100

lr = 1
epochs = 100
batch_size = 64

val_ratio = 0.11

BCELoss = nn.BCELoss()
sigmoid = nn.Sigmoid()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("[*] running on", device)

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user)*self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)

def acc_fn(logit, label):
    logit = logit.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    pred = (logit>0).astype(np.int16)
    score = (pred==label).mean()
    return score

def main(train_path):
    df = pd.read_csv(train_path)
    users = df['UserId'].values
    items = df['ItemId'].values
    n_user = len(users)

    p_pairs = []
    n_pairs = []
    
    uilist = [ [] for i in range(n_user) ]
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

    p_pairs = np.array(p_pairs)
    n_pairs = np.array(n_pairs)
    
    np.random.shuffle(p_pairs)
    np.random.shuffle(n_pairs)
    
    val_n = int(len(p_pairs)*val_ratio)
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
    val_dataset = TensorDataset(val_x, val_y)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size)
    val_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size)

    model = MatrixFactorization(n_user, n_item, n_factors=n_factor)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.zero_grad()
    
    print("[*] start training")
    # train
    model.train()
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            users = x_batch[:, 0] 
            items = x_batch[:, 1] 
            y_pred = model(users, items)

            loss = loss_fn(y_pred, y_batch)
            acc = acc_fn(y_pred, y_batch)
            
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc

        #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_dataloader):.5f}')
       
        # eval
        model.eval()
        epoch_acc_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                users = x_batch[:, 0] 
                items = x_batch[:, 1] 
                y_test_pred = model(users, items)
                acc = acc_fn(y_test_pred, y_batch)
                epoch_acc_val += acc

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_dataloader):.5f} | Acc: {epoch_acc/len(train_dataloader):.3f} | VAL Acc: {epoch_acc_val/len(train_dataloader):.3f}')

if __name__ == "__main__":
    main('train.csv')
