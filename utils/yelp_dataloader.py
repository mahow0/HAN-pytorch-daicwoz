import sys
#sys.path.insert(1, '/content/HAN-pytorch-daicwoz/utils')
#sys.path.insert(1, '/content/HAN-pytorch-daicwoz/src')
sys.path.insert(1, '../src/')

from dataloader import get_dataloader
from yelp_dataset import Yelp_Dataset
import gensim


def get_train_val_test_split(path_to_json : str, total_documents : int, split : list, train_params : dict, eval_params : dict, model : gensim.models.keyedvectors.KeyedVectors):

    assert len(split) == 3
    print('Loading Yelp Dataset')
    dataset = Yelp_Dataset(path_to_json, max_documents=total_documents)
    print("Done")
    train, val, test = dataset.split_dataset(split)

    train_loader = get_dataloader(train, train_params, model)
    val_loader = get_dataloader(val, eval_params, model)
    test_loader = get_dataloader(test, eval_params, model)

    return train_loader, val_loader, test_loader
    


