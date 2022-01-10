from dataset import HAN_Dataset
import json
import re
import gc
from math import floor
from tqdm import tqdm
from pathlib import Path
from icecream import ic
import pickle as pkl
import pandas as pd
import numpy as np



class Yelp_Dataset(HAN_Dataset):

    
    def __init__(self, path_to_json, max_documents = -1, chunksize = 10000, filterlen=0):
        
        super().__init__()

        #Set paths
        self.path_to_json = path_to_json
        

        #Extract reviews
        field_dtypes = {"stars": np.float16, 
                            "useful": np.int32, 
                            "funny": np.int32,
                            "cool": np.int32}

        count = 0   
        with pd.read_json(path_to_json, 
                                    orient='records', 
                                    lines = True, 
                                    dtype = field_dtypes,
                                    chunksize = chunksize
                                    ) as review_table:

            for chunk in tqdm(review_table):

                if max_documents != -1 and count >= max_documents:
                        break

                for review in chunk.itertuples(index=False):

                    if max_documents != -1 and count >= max_documents:
                        break

                    text : str = review.text

                    #Remove most whitespace characters
                    text = re.sub('[\t\n\r\f\v]+', '', text)
                    #Convert to lower case and then remove trailing and leading whitespaces
                    text = text.lower().strip()
                
                    #Split by punctuation and strip each individual piece
                    sentences = [s.strip() for s in re.split('[?.,!]', text)]

                    if len(sentences[-1]) == 0:
                        sentences = sentences[:-1]

                    #Filter by length
                    if filterlen != 0 and len(sentences) > filterlen:
                        continue
                    
                    #Get star rating
                    label : int = floor(review.stars) - 1

                    self.documents.append(sentences)
                    self.labels.append(label)

                    count += 1

        self.max_documents = len(self.documents)

    def split_dataset(self, split):

        assert sum(split) == 1
        sets = []
        pointer = 0
        for i, fraction in enumerate(split):

            dataset = HAN_Dataset()
            dataset.documents = self.documents[pointer : pointer + floor(fraction*self.max_documents)] if i != len(split) - 1 else self.documents[pointer :]
            dataset.labels = self.labels[pointer : pointer + floor(fraction*self.max_documents)] if i != len(split) - 1 else self.labels[pointer :]
            sets.append(dataset)

            pointer += floor(fraction*self.max_documents)
        
        return sets

        




            










            
if __name__ == '__main__':
    '''
    path_to_json = r'/home/mahmoud/Documents/nlp-depression-detection-suite/data/yelp/yelp_dataset/yelp_academic_dataset_review.json'
    dataset = Yelp_Dataset(path_to_json, 100000)
    print(len(dataset))
    print(dataset[5])
    '''
    path_to_json = r'/home/mahmoud/Documents/nlp-depression-detection-suite/data/yelp/yelp_dataset/yelp_academic_dataset_review.json'
    dataset = Yelp_Dataset(path_to_json, max_documents=500000, chunksize=10000)
    print(len(dataset))


