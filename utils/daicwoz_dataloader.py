from src.daicwoz_dataset import DAICWOZ_Dataset
from utils.dataloader import get_dataloader


def get_daicwoz_dataset(csv_path, transcript_path, dataloader_params, model, label_name = 'PHQ8_Binary', oversample_minority = False):
    dataset = DAICWOZ_Dataset(csv_path, transcript_path, label_name=label_name, oversample_minority=oversample_minority)
    return dataset

def get_daicwoz_dataloader(csv_path, transcript_path, dataloader_params, model, label_name = 'PHQ8_Binary', oversample_minority = False):

    dataset = DAICWOZ_Dataset(csv_path, transcript_path, label_name = label_name, oversample_minority=oversample_minority)
    dataloader = get_dataloader(dataset, dataloader_params, model)

    return dataloader
