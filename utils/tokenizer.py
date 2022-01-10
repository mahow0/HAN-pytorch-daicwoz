import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
import gensim.downloader
from icecream import ic

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

def get_index(model, word):

    #Pad with 0 if the word is not in the model's vocabulary
    if word not in model.key_to_index.keys():
        return 0

    return model.key_to_index[word]

def tokenize(model : gensim.models.keyedvectors.KeyedVectors, sentence):

    tokenized_sentence = []
    for word in sentence.split(' '):
        tokenized_sentence.append(get_index(model,word))

    return tokenized_sentence

def tokenize_document(model : gensim.models.keyedvectors.KeyedVectors, document):

    tokenized_document = []
    for sentence in document:
        sent = tokenize(model, sentence)
        sent = torch.LongTensor(sent)
        tokenized_document.append(sent)


    return tokenized_document

def collate_documents(model : gensim.models.keyedvectors.KeyedVectors, documents, labels):

    tokenized_documents = []
    max_num_words = 0
    max_num_sentences = 0

    for document in documents:
        doc = tokenize_document(model, document)

        num_words = max([len(_) for _ in doc])
        num_sentences = len(doc)

        if num_words > max_num_words:
            max_num_words = num_words

        if num_sentences > max_num_sentences:
            max_num_sentences = num_sentences

        doc = nn.utils.rnn.pad_sequence(doc, batch_first=True)
        tokenized_documents.append(doc)

    temp = []
    for doc in tokenized_documents:

        n_words = doc.size(1)
        p1d = (0, max_num_words-n_words)
        new_doc = F.pad(doc, p1d)
        temp.append(new_doc)

    doc_tensor = nn.utils.rnn.pad_sequence(temp, batch_first=True).int()
    attention_mask = generate_doc_attention_masks(doc_tensor).bool()
    return doc_tensor, attention_mask, labels

def generate_doc_attention_masks(doc_tensor):
    return (doc_tensor != 0)


if __name__ == '__main__':
    word2vec = get_pretrained_word2vec()
    documents = [['hello my name is', 'hello world'], ['what is this', 'how are you on this day today', 'okay so how about']]

    doc_tensor, attention_mask = collate_documents(word2vec, documents)
    #ic(doc_tensor.size())
    #for i in doc_tensor.permute(1,0,2):
        #ic(i)
    attention_mask = generate_doc_attention_masks(doc_tensor)
    ic(attention_mask)
    ic(torch.max(attention_mask,dim=2)[0])









