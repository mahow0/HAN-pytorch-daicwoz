import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy, Precision, Recall
from tqdm import tqdm
from icecream import ic

def evaluate(
    model: nn.Module,
    test_set: DataLoader,
    device = torch.device('cpu')
):
    softmax = nn.Softmax()
    model.eval()
    acc = Accuracy()
    precision = Precision(average=False)
    recall = Recall(average=False)
    f1 = (precision * recall * 2 / (precision + recall)).mean()
    batch_size = test_set.batch_size 

    for item in tqdm(test_set):

        input_ids, attention_mask, labels = item
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = torch.Tensor(labels).to(device)

        with torch.no_grad():
          # Feed the batch into the model
          output = model(input_ids, attention_mask)
          effective_batch_size = output.size(0)

          #ic(output.size())
          #ic(softmax(output).size())
          output = softmax(output.squeeze(1)).max(dim=1, keepdim=True)[1].long()
          #ic(output.size())
          if effective_batch_size < batch_size:
            batch_size = effective_batch_size
          labels = labels.reshape((batch_size,)).long()
          #ic(output)
          #ic(labels)
          acc.update((output, labels))
          precision.update((output, labels))
          recall.update((output, labels))
          f1.update((output, labels))

    acc = acc.compute()
    precision = precision.compute()
    recall = recall.compute()
    f1 = f1.compute()

    return acc, precision, recall, f1