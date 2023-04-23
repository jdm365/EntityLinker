import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from jarowinkler import jarowinkler_similarity
import numpy as np
from tqdm import tqdm
import logging




class JaroModel(nn.Module):
    def __init__(self, max_len=256, emb_dim=512, lr=1e-3):
        super(JaroModel, self).__init__()
        self.model = nn.Sequential(
                nn.Embedding(num_embeddings=max_len, embedding_dim=emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, X):
        X = self.model(X)
        X = X.mean(dim=-2)
        return X


    def get_embedding(self, str):
        str_tensor = T.tensor([ord(char) for char in str.lower()], dtype=T.long)
        str_tensor = str_tensor.to(self.device)
        return self.forward(str_tensor)

    def save(self, filename='trained_models/jaro_model.pt'):
        logging.info(f'Saving model to {filename}')
        T.save(self.state_dict(), filename)


    def load(self, filename='trained_models/jaro_model.pt'):
        logging.info(f'Loading model from {filename}')
        self.load_state_dict(T.load(filename))
        self.to(self.device)




class JaroDataset(Dataset):
    def __init__(self, str_list, sim_func):
        self.str_list = str_list
        self.sim_func = sim_func

        ## Max length of string
        max_len = 0
        for str in str_list:
            if len(str) > max_len:
                max_len = len(str)
        self.max_len = max_len + (8 - max_len % 8)


    def __len__(self):
        ## Length is arbitrary here, but length of one of the list
        return len(self.str_list)

    def __getitem__(self, idx):
        other_idx = np.random.randint(0, len(self.str_list))
        str1 = self.str_list[idx]
        str2 = self.str_list[other_idx]
        sim_val = 2 * self.sim_func(str1, str2) - 1

        ## Convert to utf8 lowercase int Tensor
        str1_tensor = T.tensor([ord(char) for char in str1.lower()], dtype=T.long)
        str2_tensor = T.tensor([ord(char) for char in str2.lower()], dtype=T.long)

        ## Pad to max_len
        str1_tensor = F.pad(str1_tensor, (0, self.max_len - len(str1_tensor)))
        str2_tensor = F.pad(str2_tensor, (0, self.max_len - len(str2_tensor)))

        return str1_tensor, str2_tensor, sim_val


class CustomCosineEmbeddingLoss(nn.Module):
    def __init__(self):
        super(CustomCosineEmbeddingLoss, self).__init__()

    def forward(self, pred1, pred2, sim_val):
        loss = (F.cosine_similarity(pred1, pred2, dim=-1) - sim_val).pow(2).mean()
        return loss



def train(
        str_list, 
        n_epochs=4, 
        emb_dim=512, 
        lr=1e-3, 
        batch_size=128, 
        num_workers=4,
        filename='trained_models/jaro_model.pt'
        ):
    dataloader = DataLoader(
            JaroDataset(str_list, jarowinkler_similarity),
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=True
            )
    model = JaroModel(
            emb_dim=emb_dim,
            lr=lr
            )
    loss_fn = CustomCosineEmbeddingLoss()
    
    progress_bar = tqdm(total=n_epochs * len(dataloader))

    for epoch in range(n_epochs):
        for _, (str1, str2, sim_val) in enumerate(dataloader):
            str1    = str1.to(model.device)
            str2    = str2.to(model.device)
            sim_val = sim_val.to(model.device)

            pred1 = model.forward(str1)
            pred2 = model.forward(str2)
            loss = loss_fn(pred1, pred2, sim_val)

            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            if loss.item() < 0.02:
                break

            progress_bar.update(1)
            progress_bar.set_description(f'Epoch: {epoch}, Loss: {loss.item()}')

    progress_bar.close()

    model.save(filename=filename)
    return model


def get_embeddings(str_list, model):
    with T.no_grad():
        embeddings = []
        for x in tqdm(str_list, desc='Getting embeddings'):
            embeddings.append(model.get_embedding(x))

        embeddings = T.stack(embeddings).cpu().detach().numpy()
    return embeddings



def train_and_get_embeddings(str_list, **kwargs):
    model = train(str_list, **kwargs)
    embeddings = get_embeddings(str_list, model)
    return embeddings
