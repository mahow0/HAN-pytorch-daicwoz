import sys
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/utils')
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/src')
from main import * 
import numpy as np

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
  results = []
  for hidden_size in range(hidden_size_begin, hidden_size_end, hidden_size_skip):
    for lr in np.arange(lr_begin, lr_end, lr_skip):

      model_config = {'num_classes':2, 'embed_dim':embedding_model.vector_size, 'hidden_dim': hidden_size, 'attn_dim': hidden_size, 'num_layers':2, 'dropout':0.2, 'embedder': embedder}
      #print(eval_set.dataset[0])
      optim_params = {'lr': lr}
      acc, precision, recall, f1 = main(num_epochs, optim_params, train_set, eval_set, val_set, model_config, device=device)
      profile = {'hidden_dim':hidden_size, 'lr': lr, 'acc':acc, 'precision':precision, 'recall':recall, 'f1':f1}
      results.append(profile)
  
  return results



if __name__ == '__main__':
  
  results = grid_search(100, 300, 0.05, 0.5)
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


