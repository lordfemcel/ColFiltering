import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Engine(object):
    def __init__(self, config):
        self.model = None
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.criterion = nn.BCELoss()
        if config['optimizer'] == 'adam':
            self.optimizer = lambda model: optim.Adam(model.parameters(), lr=config['adam_lr'], weight_decay=config['l2_regularization'])
        else:
            self.optimizer = lambda model: optim.SGD(model.parameters(), lr=config['adam_lr'], weight_decay=config['l2_regularization'])
        self.config = config


    def train_an_epoch(self, train_loader, epoch):
        self.model.train()
        optimizer = self.optimizer(self.model)
        total_loss = 0
        for user, item, label in train_loader:
            user = user.view(-1).to(self.device)
            item = item.view(-1).to(self.device)
            label = label.view(-1).to(self.device)
            self.model.zero_grad()
            prediction = self.model(user, item)
            loss = self.criterion(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            
        print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")
        
       
    def evaluate(self, evaluate_data, epoch, k=10):
        self.model.eval()
        test_users, test_items, neg_users, neg_items = evaluate_data
        hits, ndcgs = [], []
        num_users = len(test_users)
        with torch.no_grad():
            for idx in range(num_users):
                u = test_users[idx].item()
                gtItem = test_items[idx].item()
                # 99 negative + 1 positive
                item_candidates = [gtItem]
                # 99 negatives for this user
                item_candidates += [neg_items[i].item() for i in range(idx*99, (idx+1)*99)]
                users = torch.LongTensor([u]*100).to(self.device)
                items = torch.LongTensor(item_candidates).to(self.device)
                predictions = self.model(users, items)
                _, indices = torch.topk(predictions, k)
                recommends = torch.take(torch.tensor(item_candidates), indices.cpu())
                hr = int(gtItem in recommends)
                if hr:
                    rank = (recommends == gtItem).nonzero(as_tuple=True)[0].item() + 1
                    ndcg = np.log(2) / np.log(rank + 1)
                else:
                    ndcg = 0
                hits.append(hr)
                ndcgs.append(ndcg)
        hr_avg = np.mean(hits)
        ndcg_avg = np.mean(ndcgs)
        print(f"Epoch {epoch} HR@{k}: {hr_avg:.4f}, NDCG@{k}: {ndcg_avg:.4f}")
        return hr_avg, ndcg_avg

    def save(self, alias, epoch, hr, ndcg):
        path = f"{alias}_epoch{epoch}_hr{hr:.4f}_ndcg{ndcg:.4f}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
