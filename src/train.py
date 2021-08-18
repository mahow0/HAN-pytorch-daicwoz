import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
from evaluate import evaluate
from icecream import ic


def train(
        model: nn.Module,
        training_set: DataLoader,
        validation_set: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        num_epochs: int,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device('cpu')
):
    # Set the model to training mode and fix the upper bound on gradient norm
    model.train()
    max_grad_norm = 1

    # Obtain the number of training examples
    num_batches = len(training_set)

    for _ in range(num_epochs):

        training_set = tqdm(training_set, desc=f"Epoch: {_ + 1}/{num_epochs}")
        losses = []

        for current_batch, batch in enumerate(training_set):

            # Separate the batch into input_ids and attention_mask
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = torch.Tensor(labels).to(device)
            #ic(input_ids.size())
            #ic(input_ids.size())
            #ic(attention_mask.size())
            #ic(labels.size())
            # print(input_ids.size())
            # print(attention_mask.size())
            # print(labels.size())

            # Zero gradients for the model parameters, does the same thing as optimizer.zero_grad()
            model.zero_grad()

            # Feed the batch into the model
            output = model(input_ids, attention_mask)
            # Calculate and backpropagate loss, clip gradient norm
            output = output.squeeze(1)
            loss = loss_fn(output, labels.long())
            #ic(output)
           # ic(labels)
            #ic(loss)

            loss.backward()
            loss.detach()
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update parameters and learning rate
            optimizer.step()

            training_set.set_postfix(
                {"Batch": f"{current_batch}/{num_batches}", "Loss": loss.item()}
            )
            # Clear the GPU memory of batch that we no longer need
            input_ids.detach()
            attention_mask.detach()
            labels.detach()
            output.detach()
            del output
            del input_ids
            del attention_mask
            del labels
            del loss
            gc.collect()
            torch.cuda.empty_cache()

        acc, precision, recall, f1 = evaluate(model, validation_set, device=device)
        print(f'Accuracy: {acc}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
        model.train()
        scheduler.step(acc)

    return model