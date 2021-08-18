from dataset import HAN_Dataset
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from pathlib import Path
import pickle as pkl

def oversample(dataframe, name, label, strategy='minority'):
    oversampler = RandomOverSampler(sampling_strategy=strategy)

    participants = dataframe[name].values.reshape(-1, 1)
    labels = dataframe[label].values

    participants_over, labels_over = oversampler.fit_resample(participants, labels)
    return participants_over, labels_over


class DAICWOZ_Dataset(HAN_Dataset):
    def __init__(
        self, path_to_csv, path_to_transcripts, filterlen=0, label_name="PHQ8_Binary",
    oversample_minority = True):
        super().__init__()

        # Set paths
        self.path_to_csv = path_to_csv
        self.path_to_transcripts = path_to_transcripts

        # Initialize dataset
        self.oversample = oversample_minority

        # Grab a list of all the speakers
        csv = pd.read_csv(self.path_to_csv)
        speakers = csv["Participant_ID"].values

        temp_transcripts = []
        temp_labels = []

        self.num_depressed = 0
        self.num_undepressed = 0
        if self.oversample:
            speakers, _ = oversample(csv, 'Participant_ID', label_name)
            speakers = speakers.reshape(-1)

        # Iterate through the dataset and store the participants' responses in a list
        for speaker in tqdm(speakers):
            transcript_file = Path(path_to_transcripts) / f"{speaker}_TRANSCRIPT.csv"
            transcript_df = pd.read_csv(transcript_file, sep="\t")
            # remove beginning and end spaces
            transcript_df.value = transcript_df.value.str.strip()
            transcript_df.dropna(inplace=True)

            # remove entries below length specified by args.filterlen
            transcript_df = transcript_df[
                transcript_df.value.str.split().apply(len) > filterlen
            ]
            transcript_df = transcript_df[transcript_df.speaker == "Participant"]

            # take out possible filler material at the beginning
            transcript_df = list(transcript_df["value"].values)
            if transcript_df[0] in ["<synch>", "<sync>", "[syncing]"]:
                transcript_df = transcript_df[1:]

            label = csv[csv["Participant_ID"] == speaker][label_name].values.item()
            if label == 0:
                self.num_undepressed += 1
            else:
                self.num_depressed += 1
            temp_transcripts.append(transcript_df)
            temp_labels.append(label)

        self.documents = temp_transcripts
        self.labels = temp_labels

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pkl.dump(self, file)



if __name__ == '__main__':
    path_to_train = r'E:\depression_text\LABELS\train_split_Depression_AVEC2017.csv'
    path_to_transcripts = r'E:\depression_text\TRANSCRIPT'
    #dataset = DAICWOZ_Dataset(path_to_train, path_to_transcripts)
    #dataset.save('train.pkl')





