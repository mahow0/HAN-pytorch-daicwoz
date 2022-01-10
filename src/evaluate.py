import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy, Precision, Recall
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from icecream import ic

def evaluate(
    model: nn.Module,
    test_set: DataLoader,
    device = torch.device('cpu')
):
    softmax = nn.Softmax()
    model.eval()
    batch_size = test_set.batch_size 
    output_batches = []
    label_batches = []

    for item in tqdm(test_set):

        input_ids, attention_mask, labels = item
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = torch.Tensor(labels).to(device)

        with torch.no_grad():
          # Feed the batch into the model
          output = model(input_ids, attention_mask)
          effective_batch_size = output.size(0)

          #ic(softmax(output).size())
          output = softmax(output.squeeze(1)).max(dim=1, keepdim=True)[1].long()
          output_batches.append(output)

          if effective_batch_size < batch_size:
            batch_size = effective_batch_size

          labels = labels.reshape((batch_size, 1)).long()
          label_batches.append(labels)
          #ic(output)
          #ic(labels)
 
 
    outputs = torch.cat(output_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    report = classification_report(labels, outputs, output_dict=True)
    report_str = classification_report(labels, outputs, output_dict=False)

    ic(report_str)

    acc = report['accuracy']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']

    return acc, precision, recall, f1


if __name__ == '__main__':
  output = np.array([0, 1, 2]).reshape(3, 1)
  labels = np.array([1, 1, 1]).reshape(3, 1)

  report = classification_report(labels, output, output_dict = True)
  print(report)
