from torch.utils.data import Dataset


class HAN_Dataset(Dataset):

    def __init__(self):
        self.documents = []
        self.labels = []

    def __len__(self):
        assert len(self.documents) == len(self.labels)
        return len(self.documents)

    def __getitem__(self, i):
        return self.documents[i], self.labels[i]

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pkl.dump(self, file)

