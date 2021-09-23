import sys
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/utils')
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/src')
import torch
from train import * 
import numpy as np
from daicwoz_dataloader import get_daicwoz_dataloader, get_daicwoz_dataset
from embeddings import get_pretrained_word2vec, get_pretrained_glove, GensimEmbedder
from model import HAN
import gc

train_csv_path = r'./drive/MyDrive/depression_text/LABELS/train_split_Depression_AVEC2017.csv'
val_csv_path = r'./drive/MyDrive/depression_text/LABELS/dev_split_Depression_AVEC2017.csv'
eval_csv_path =  r'./drive/MyDrive/depression_text/LABELS/full_test_split.csv' 
transcript_path = r'./drive/MyDrive/depression_text/TRANSCRIPT'

def grid_search(hidden_size_begin, hidden_size_end,lr_begin, lr_end, hidden_size_skip = 50, lr_skip = 0.01, num_epochs=20, cuda=True):

  embedding_model = get_pretrained_word2vec()
    #embedding_model = get_pretrained_glove()
  if cuda:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
      device = torch.device("cpu")

  print(device)
  train_set_params = {'batch_size': 8, 'shuffle': True, 'num_workers': 8, 'pin_memory':True}
  eval_set_params = {'batch_size':8, 'num_workers': 8}

  embedder = GensimEmbedder(embedding_model).to(device)

  train_set = get_daicwoz_dataloader(train_csv_path, transcript_path, train_set_params, embedding_model, oversample_minority=True)
  val_set =  get_daicwoz_dataloader(val_csv_path, transcript_path, eval_set_params, embedding_model)
  eval_set =  get_daicwoz_dataloader(eval_csv_path, transcript_path, eval_set_params, embedding_model, label_name='PHQ_Binary')

  num_depressed = train_set.dataset.num_depressed
  num_undepressed = train_set.dataset.num_undepressed
  total = num_depressed + num_undepressed
  class_weights = torch.Tensor([1, 1]).to(device)

  cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

  results = []
  for hidden_size in range(hidden_size_begin, hidden_size_end, hidden_size_skip):

    for lr in np.arange(lr_begin, lr_end, lr_skip):
    
      model_config = {'num_classes':2, 'embed_dim':embedding_model.vector_size, 'hidden_dim': hidden_size, 'attn_dim': hidden_size, 'num_layers':2, 'dropout':0.2, 'embedder': embedder}
      optim_params = {'lr': lr}

      #Intialize model and transformer
      model = HAN(**model_config).to(device)
      #Initialize optimizer and scheduler
      optimizer = optim.Adam(model.parameters(), **optim_params)
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, cooldown=1, verbose=False, threshold=0.001)
      #print(eval_set.dataset[0])
      model, epoch_profiles = train(model, train_set, val_set, optimizer = optimizer, scheduler=scheduler, num_epochs = num_epochs, device=device, loss_fn = cross_entropy, grid_search = True)

      max_acc = sorted(epoch_profiles, key = lambda s : s['acc'])[::-1][0]
      max_precision = sorted(epoch_profiles, key = lambda s : s['precision'])[::-1][0]
      max_recall = sorted(epoch_profiles, key = lambda s : s['recall'])[::-1][0]
      max_f1 = sorted(epoch_profiles, key = lambda s : s['f1'])[::-1][0]

      acc = max_acc['acc']
      precision = max_precision['precision']
      recall = max_recall['recall']
      f1 = max_f1['f1']


      profile = {'hidden_dim':hidden_size, 'lr': lr, 'acc':acc, 'precision':precision, 'recall':recall, 'f1':f1}
      results.append(profile)
      torch.save(model.state_dict(), f'/content/drive/MyDrive/HAN-logs/hdim_{hidden_size}_lr_{lr}.pt')

      del model
      gc.collect()
      torch.cuda.empty_cache()
      file_name = f'HAN_log.txt'
      summary_lines = [f'Hidden size: {hidden_size}\n', f'Learning rate: {lr}\n', f'Max acc: {acc}\n', f'Max precision: {precision}\n', f'Max recall: {recall}\n', f'Max f1: {f1}\n', '\n\n']
      detailed_stats = [f'Max acc profile: {max_acc}\n', f'Max precision profile: {max_precision}\n', f'Max recall profile: {max_recall}\n', f'Max f1 profile: {max_f1}\n']
      with open(f'/content/drive/MyDrive/HAN-logs/{file_name}', 'a') as f:
        f.writelines(lines)

  
  return results



if __name__ == '__main__':
  
  results = grid_search(64, 128, 0.001, 0.08)
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


