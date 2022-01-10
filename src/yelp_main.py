#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys, os
path = os.getcwd()
sys.path.insert(1, os.path.join(path, 'utils'))
sys.path.insert(1, os.path.join(path, 'src'))

from evaluate import evaluate
from train import *
from transformers import AutoTokenizer
from yelp_dataloader import get_train_val_test_split
from embeddings import get_pretrained_word2vec, get_pretrained_glove, GensimEmbedder
from model import HAN
from tokenizer import collate_documents

import torch
import torch.nn as nn
from icecream import ic
import argparse


def collate_fn(embedding_model, batch):
    documents = [_[0] for _ in batch]
    labels = [_[1] for _ in batch]

    return collate_documents(embedding_model, documents, labels)

def main( num_epochs : int, optim_params : dict, train_set, test_set, val_set, model_config,  save_dir : str = './', device=torch.device('cpu')):

    #Intialize model and transformer
    model = HAN(**model_config).to(device)
    #print(model)

    #Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), **optim_params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, cooldown=1, verbose=False, threshold=0.001)

    #num_depressed = train_set.dataset.num_depressed
    #num_undepressed = train_set.dataset.num_undepressed
    #total = num_depressed + num_undepressed
    #class_weights = torch.Tensor([1, 1]).to(device)

    #cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
    cross_entropy = nn.CrossEntropyLoss()

    #Train model
    torch.cuda.empty_cache()
    model = train(model, train_set, val_set, optimizer = optimizer, scheduler=scheduler, num_epochs = num_epochs, device=device, loss_fn = cross_entropy)

    #Save model weights
    checkpoint_name = f'HAN_{num_epochs}epochs.pt'
    save_dir = save_dir + checkpoint_name
    torch.save(model.state_dict(), save_dir)

    #Evaluate model
    acc, precision, recall, f1 = evaluate(model, test_set, device=device)

    ic(f'Accuracy: {acc}')
    ic(f'Precision: {precision}')
    ic(f'Recall: {recall}')
    ic(f'F1: {f1}')

    return acc, precision, recall, f1



if __name__ == '__main__':

    #Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--yelp_reviews', type = str, help= 'path to json file')
    parser.add_argument('--total_reviews', type = int, default = 50000)
    parser.add_argument('--lr', type = float, nargs='?', default = 0.005, help='Learning rate')
    parser.add_argument('--num_epochs', type = int, nargs= '?', default = 10, help = 'Number of epochs')
    parser.add_argument('--cuda', type = bool,nargs='?', default=True, help='Enable GPU acceleration if available')

    args = parser.parse_args()
    ic('Loading pretrained Word2Vec model...')
    embedding_model = get_pretrained_word2vec()
    #embedding_model = get_pretrained_glove()
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    ic(device)
    train_set_params = {'batch_size': 64, 'shuffle': True, 'num_workers': 4, 'pin_memory':False}
    eval_set_params = {'batch_size':8, 'num_workers': 4}

    embedder = GensimEmbedder(embedding_model).to(device)
    model_config = {'num_classes':5, 'embed_dim':embedding_model.vector_size, 'hidden_dim': 50, 'attn_dim':100, 'num_layers':1, 'dropout':0.25, 'embedder': embedder}

    split = [0.8, 0.1, 0.1]
    ic('Preparing dataset...')
    ic(args.yelp_reviews)
    ic(args.total_reviews)
    train_set, val_set, test_set = get_train_val_test_split(args.yelp_reviews, args.total_reviews, split, train_set_params, eval_set_params, embedding_model)
    ic(len(train_set))
    ic(len(val_set))
    ic(len(test_set))

    '''
    train_set = get_daicwoz_dataset(args.train_csv_path, args.transcript_path, train_set_params, embedding_model)
    train_set = DataLoader(dataset=train_set, collate_fn=collate_fn, **train_set_params)
    val_set =  get_daicwoz_dataset(args.val_csv_path, args.transcript_path, eval_set_params, embedding_model)
    val_set = DataLoader(dataset=val_set, collate_fn=collate_fn, **eval_set_params)
    eval_set =  get_daicwoz_dataset(args.eval_csv_path, args.transcript_path, eval_set_params, embedding_model, label_name='PHQ_Binary')
    eval_set = DataLoader(dataset=eval_set, collate_fn=collate_fn, **eval_set_params)
    '''
    #print(eval_set.dataset[0])

    optim_params = {'lr': args.lr}
    acc, precision, recall, f1 = main(args.num_epochs, optim_params, train_set, test_set, val_set, model_config, device=device)



#TODO: Chain together val and eval sets
#TODO: Allow main.py to interface with a file to read off config parameters
