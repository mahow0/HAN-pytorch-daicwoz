import sys
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/utils')
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/src')


import gensim.models.keyedvectors
from dataset import HAN_Dataset
from tokenizer import collate_documents
from torch.utils.data import DataLoader

def get_dataloader(dataset : HAN_Dataset, data_params : dict, model : gensim.models.keyedvectors.KeyedVectors):

    def collate_fn(batch):
        documents = [_[0] for _ in batch]
        labels = [_[1] for _ in batch]

        return collate_documents(model, documents, labels)

    return DataLoader(dataset, collate_fn=collate_fn, **data_params)



