import sys
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/utils')
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/src')

import torch
import torch.nn as nn
import gensim
import gensim.downloader
import tokenizer
import numpy as np
from icecream import ic

class GensimEmbedder(nn.Module):

    def __init__(self, model : gensim.models.keyedvectors.KeyedVectors):
        super(GensimEmbedder, self).__init__()
        weights = torch.from_numpy(model.vectors).float()
        self.embed = nn.Embedding.from_pretrained(weights)
        self.embed_dim = model.vector_size
        self.zero_index_vector = self.embed(torch.LongTensor([0]))
        self.zero_vector = torch.zeros((1, self.embed_dim))

    def to(self, device):
        self.zero_index_vector = self.zero_index_vector.to(device)
        self.zero_vector = self.zero_vector.to(device)

        return super(GensimEmbedder, self).to(device)

    def forward(self, x):
        '''

        :param x: (torch.Tensor) Token ids of sentence. Shape: (Batch, Max Word Length)
        :return: (torch.Tensor) Embeddings for that sentence. Shape: (Batch, Max Word Length, Embed Dim)
        '''
        x = self.embed(x,)
        x[torch.all(x == self.zero_index_vector, dim=2)] = self.zero_vector
        return x

def get_pretrained_word2vec():
    '''
    Retrieves pretrained Word2Vec model from gensim
    :return: (gensim.models.keyedvectors.KeyedVectors) Pretrained Word2Vec model
    '''
    return gensim.downloader.load('word2vec-google-news-300')

def get_pretrained_glove():
    '''
    Retrieves pretrained GloVe model from gensim
    :return: (gensim.models.keyedvectors.KeyedVectors) Pretrained GloVe model
    '''
    return gensim.downloader.load('glove-wiki-gigaword-300')

def embed(model : gensim.models.keyedvectors.KeyedVectors , sentences : list):
    '''
    Uses a pretrained model to embed a sequence of sentences
    :param model: (gensim.models.keyedvectors.KeyedVectors) Pretrained model used to compute the embeddings
    :param sentences: (str[]) Sequence of sentences, representing one document
    :return: embeddings: (torch.Tensor[]) Word-level embeddings
             lengths: (int[]) List consisting of the lengths of each batch
    '''

    embeddings = []
    lengths = []
    dim = model.vector_size
    for sentence in sentences:
        sentence_feature = []

        for word in sentence.split(' '):

            if word in model:
                vec = torch.from_numpy(model[word.lower()])

            if word not in model:
                vec = torch.zeros((dim,))

            sentence_feature.append(vec)

        sentence_feature = torch.stack(sentence_feature)
        embeddings.append(sentence_feature)
        new_length = len(sentence.split())
        lengths.append(new_length)


    return embeddings, lengths


def pad_embeddings(embeddings):
    '''
    Pad batches of embeddings to max length
    :param embeddings: (torch.Tensor[]) Iterable of embedding sequences, each of shape (*, hidden_dim)
    :return: (torch.Tensor) Padded embedding tensor. Shape: (Batch, Max Length, hidden_dim)
    '''
    return torch.nn.utils.rnn.pad_sequence(embeddings,batch_first=True)

def generate_mask(embeddings, lengths):
    '''
    Generates mask used for attention. Padded vectors are masked out.
    :param embeddings: (torch.Tensor) Padded embedding tensor. Shape: (Batch, Max Length, hidden_dim)
    :param lengths: (int[]) List consisting of the (unpadded) lengths of each batch
    :return: mask: (torch.Tensor) Mask. Shape: (Batch, Max Length)
    '''

    max_length = embeddings.size(1)
    tensor_lengths = torch.Tensor(lengths)
    mask = (torch.arange(max_length)[None, :] < tensor_lengths[:, None]).long()
    return mask



'''
def collate_documents(model : gensim.models.keyedvectors.KeyedVectors, documents : list):


    :param documents: ((str[])[]) List of documents
    :param model : (gensim.models.keyedvectors.KeyedVectors) Pretrained model used to compute embedding
    :return:
    

    word_lengths = []
    word_masks = []

    word_embeddings = []
    for document in documents:

        #Embed all the words in the document
        embeddings, lengths = embed(model, document)

        #Pad to maximum word-length in that document
        embeddings = pad_embeddings(embeddings)

        # Generate attention mask
        mask = generate_mask(embeddings, lengths)
        ic(mask.size())

        #Resize so that final tensor's batch size reflects the number of documents
        dim = embeddings.size(2)
        embeddings = embeddings.view(1, -1, dim).squeeze(0)


        lengths = torch.LongTensor(lengths)
        word_embeddings.append(embeddings)
        word_lengths.append(lengths)
        word_masks.append(mask)

    sentence_embeddings = pad_embeddings(word_embeddings)
    sentence_lengths = word_lengths
    sentence_masks = word_masks

    return sentence_embeddings, sentence_lengths, sentence_masks
'''


def collate_documents(model : gensim.models.keyedvectors.KeyedVectors, documents : list):
    pass

def get_index(model, word):
    return model.key_to_index[word]


if __name__ == '__main__':

    word2vec = get_pretrained_word2vec()
    embedder = GensimEmbedder(word2vec)
    documents = [['hello my name is', 'hello world'],
                 ['what is this', 'how are you on this day today', 'okay so how about']]

    doc_tensor, attention_mask = tokenizer.collate_documents(word2vec, documents)
    doc_tensor = doc_tensor.permute(1,0,2)
    doc_tensor = doc_tensor[0, :]
    #ic(doc_tensor.size())
    doc_tensor = embedder(doc_tensor)
    ic(doc_tensor.size())
    ic(doc_tensor[0, 4, :])
    #weights = torch.FloatTensor(word2vec.vectors)
    #embedding = nn.Embedding.from_pretrained(weights)
    #pad = 0
    #ic(word2vec.index_to_key[0])
    #ic(embedding(torch.LongTensor([pad])))
    #get_index(word2vec, 'banana')
    #input = torch.LongTensor([word2vec.vocab['banana'].index])

    #ic(word2vec['banana'])
    #ic(embedding(input))
    '''
    sentences = ['hello world', 'this is python code']
    embeddings, lengths = embed(word2vec, sentences)
    ic(embeddings[0].size())
    ic(lengths)
    embeddings  = pad_embeddings(embeddings)
    ic(embeddings.size())
    dim = embeddings.size(2)
    embeddings = embeddings.view(1, -1, dim)
    ic(embeddings.size())

    documents = [['hello world', 'this is python code', 'how are you', 'hi'], ['what is this', 'who is this', 'the flowers are really nice today']]
    sentence_embeddings, sentence_lengths, sentence_masks = collate_documents(word2vec, documents)
    ic(sentence_embeddings.size())
    ic(sentence_lengths)
    ic(sentence_masks)
'''
