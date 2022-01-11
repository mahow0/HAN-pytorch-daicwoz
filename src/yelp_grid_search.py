import sys, os

path = os.getcwd()
sys.path.insert(1, os.path.join(path, 'utils'))
sys.path.insert(1, os.path.join(path, 'src'))

from train import * 
import numpy as np
import argparse
from embeddings import get_pretrained_word2vec, get_pretrained_glove, GensimEmbedder
from yelp_dataloader import get_train_val_test_split
from model import HAN
import gc


def yelp_grid_search(yelp_reviews, total_reviews, hidden_size_begin, hidden_size_end,lr_begin, lr_end, hidden_size_skip = 50, lr_skip = 0.01, num_epochs=20, cuda=True):

  embedding_model = get_pretrained_word2vec()

  if cuda:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
      device = torch.device("cpu")

  print(device)
  train_set_params = {'batch_size': 64, 'shuffle': True, 'num_workers': 8, 'pin_memory':True}
  eval_set_params = {'batch_size':64, 'num_workers': 8}

  embedder = GensimEmbedder(embedding_model).to(device)

  split =[0.8, 0.1, 0.1]
  train_set, val_set, test_set = get_train_val_test_split(yelp_reviews, total_reviews, split, train_set_params, eval_set_params, embedding_model)

  cross_entropy = nn.CrossEntropyLoss()

  results = []
  for hidden_size in range(hidden_size_begin, hidden_size_end, hidden_size_skip):

    for lr in np.arange(lr_begin, lr_end, lr_skip):
    
      model_config = {'num_classes':5, 'embed_dim':embedding_model.vector_size, 'hidden_dim': hidden_size, 'attn_dim': hidden_size, 'num_layers':1, 'dropout':0.25, 'embedder': embedder}
      optim_params = {'lr': lr}

      #Intialize model and transformer
      model = HAN(**model_config).to(device)
      #Initialize optimizer and scheduler
      optimizer = optim.AdamW(model.parameters(), **optim_params)
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, cooldown=1, verbose=True, threshold=0.001)
      #print(eval_set.dataset[0])
      model, acc, precision, recall, f1 = train(model, train_set, val_set, optimizer = optimizer, scheduler=scheduler, num_epochs = num_epochs, device=device, loss_fn = cross_entropy, grid_search = True)
      profile = {'hidden_dim':hidden_size, 'lr': lr, 'acc':acc, 'precision':precision, 'recall':recall, 'f1':f1}
      results.append(profile)
      del model
      gc.collect()
      torch.cuda.empty_cache()
      file_name = f'HAN_log.txt'
      lines = [f'Hidden size: {hidden_size}\n', f'Learning rate: {lr}\n', f'acc: {acc}\n', f'precision: {precision}\n', f'recall: {recall}\n', f'f1: {f1}\n', '\n\n']
      with open(f'/content/drive/MyDrive/HAN-logs/{file_name}', 'a') as f:
        f.writelines(lines)

  
  return results



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--yelp_reviews', type=str, help='path to json file')
  parser.add_argument('--total_reviews', type=int, default=50000, help='path to json file')
  results = yelp_grid_search(yelp_reviews, total_reviews, 50, 300, 0.005, 0.08)

  max_acc = sorted(results, key= lambda s : s['acc'])[::-1][0]
  max_prec = sorted(results, key = lambda s : s['precision'])[::-1][0]
  max_recall = sorted(results, key = lambda s : s['recall'])[::-1][0]
  max_f1 = sorted(results, key = lambda s : s['f1'])[::-1][0]
  print('Best performing models:')
  accuracy = max_acc['acc']
  print(f'Best acc: {accuracy}: ')
  print(max_acc)
  precision = max_prec['precision']
  print(f'Best precision: {precision}: ')
  print(max_prec)
  recall = max_recall['recall']
  print(f'Best recall: {recall}: ')
  print(max_recall)
  f1 = max_f1['f1']
  print(f'Best f1: {f1}: ')
  print(max_f1)


