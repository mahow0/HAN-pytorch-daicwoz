#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/utils')
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/src')

from evaluate import evaluate
from train import *
from transformers import AutoTokenizer
from daicwoz_dataloader import get_daicwoz_dataloader, get_daicwoz_dataset
from embeddings import get_pretrained_word2vec, get_pretrained_glove, GensimEmbedder
from model import HAN
from tokenizer import collate_documents

import torch
import torch.nn as nn
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

    num_depressed = train_set.dataset.num_depressed
    num_undepressed = train_set.dataset.num_undepressed
    total = num_depressed + num_undepressed
    class_weights = torch.Tensor([1, 1]).to(device)

    cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    #Train model
    torch.cuda.empty_cache()
    model = train(model, train_set, val_set, optimizer = optimizer, scheduler=scheduler, num_epochs = num_epochs, device=device, loss_fn = cross_entropy)

    #Save model weights
    checkpoint_name = f'HAN_{num_epochs}epochs.pt'
    save_dir = save_dir + checkpoint_name
    torch.save(model.state_dict(), save_dir)

    #Evaluate model
    acc, precision, recall, f1 = evaluate(model, test_set, device=device)

    print(f'Accuracy: {acc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')

    return acc, precision, recall, f1



if __name__ == '__main__':

    #Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', type = str, nargs='?', required=True, help='Path to training set CSV')
    parser.add_argument('--val_csv_path', type = str, nargs='?', required=True, help='Path to validation set CSV')
    parser.add_argument('--eval_csv_path', type = str, nargs='?', required=True, help = 'Path to evaluation set CSV')
    parser.add_argument('--transcript_path', type = str, nargs='?', required=True, help='Path to folder where clinical transcripts are stored')
    parser.add_argument('--lr', type = float, nargs='?', default = 0.05, help='Learning rate')
    parser.add_argument('--num_epochs', type = int, nargs= '?', default = 40, help = 'Number of epochs')
    parser.add_argument('--cuda', type = bool,nargs='?', default=True, help='Enable GPU acceleration if available')

    args = parser.parse_args()

    embedding_model = get_pretrained_word2vec()
    #embedding_model = get_pretrained_glove()
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(device)
    train_set_params = {'batch_size': 8, 'shuffle': True, 'num_workers': 8, 'pin_memory':True}
    eval_set_params = {'batch_size':8, 'num_workers': 8}

    embedder = GensimEmbedder(embedding_model).to(device)
    model_config = {'num_classes':2, 'embed_dim':embedding_model.vector_size, 'hidden_dim': 100, 'attn_dim':100, 'num_layers':2, 'dropout':0.2, 'embedder': embedder}

    train_set = get_daicwoz_dataloader(args.train_csv_path, args.transcript_path, train_set_params, embedding_model, oversample_minority=True)
    val_set =  get_daicwoz_dataloader(args.val_csv_path, args.transcript_path, eval_set_params, embedding_model)
    eval_set =  get_daicwoz_dataloader(args.eval_csv_path, args.transcript_path, eval_set_params, embedding_model, label_name='PHQ_Binary')

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
    acc, precision, recall, f1 = main(args.num_epochs, optim_params, train_set, eval_set, val_set, model_config, device=device)



#TODO: Chain together val and eval sets
#TODO: Allow main.py to interface with a file to read off config parameters
