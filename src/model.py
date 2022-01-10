import sys
sys.path.insert(1, '/HAN-pytorch-daicwoz/utils')
sys.path.insert(1, '/content/HAN-pytorch-daicwoz/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim.downloader
from embeddings import GensimEmbedder
from tokenizer import collate_documents
from icecream import ic
import gc

class HAN(nn.Module):
    '''
    Hierarchical Attention Network. Based on paper "Hierarchical Attention Networks for Document Classification"
    by Yang et al: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
    '''

    def __init__(self, num_classes, embed_dim, hidden_dim, attn_dim, num_layers, dropout, embedder):
        super(HAN, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.sentence_attention = SentenceAttention(hidden_dim=hidden_dim, attn_dim=attn_dim)
        self.sentence_encoder = SentenceEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                                dropout=dropout, attn_dim=attn_dim, embedder=embedder)
        
        self.classifier = nn.Linear(2 * hidden_dim, num_classes, bias=True)



    def forward(self, x : torch.Tensor,  mask : torch.Tensor):
        '''
        Passes a sequence of sentences through both levels of the network
        :param x: (torch.Tensor) Padded input embeddding. Shape: (Batch, Sentence Length, Word Length)
        :param mask: (torch.Tensor) Attention mask. Shape: (Batch, Sentence Length, Word Length)
        :return: (torch.Tensor) Output classification score. Shape: (Batch, num_classes)
        '''


        document_encoding = []
        for sentence, attention_mask in zip(x.permute(1, 0, 2), mask.permute(1, 0, 2)):
            #ic(sentence.size())
            #ic(attention_mask.size())
            sent_enc = self.sentence_encoder(sentence, attention_mask)
            #ic(f'SENT_ENC: {sent_enc.size()}')
            document_encoding.append(sent_enc)

        document_encoding = torch.cat(document_encoding, dim=1)
        #ic(f'DOCUMENT_ENC:  {document_encoding.size()}')
        #Compute sentence-level mask by taking the max of row of the word dimension
        mask = torch.max(mask, dim=2)[0]
        #ic(document_encoding)
        x = self.sentence_attention(document_encoding, mask)
        #ic(x)
        x = self.classifier(x)

        return x


class SentenceAttention(nn.Module):
    '''
    Sentence-level attention mechanism
    '''

    def __init__(self, hidden_dim, attn_dim):
        super(SentenceAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.attn_sentence_linear = nn.Linear(2 * hidden_dim, self.attn_dim, bias=True)
        self.tanh = nn.Tanh()
        self.attn_sentence_context = nn.Linear(self.attn_dim, 1, bias=False)

    def forward(self, x, mask):

        seq = torch.clone(x)
        # Apply linear transformation
        x = self.tanh(self.attn_sentence_linear(x))
        #ic(f'ATTN SENT LINEAR: {x.size()}')

        # Compute dot product with context
        x = self.attn_sentence_context(x)
        #ic(f'ATTN SENT CONTEXT: {x.size()}')

        #ic(x.size())
        #ic(mask.size())
        #Mask out pad values
        x[~(mask)] = 10**(-10)

        # Compute attention weights
        x = F.softmax(x, dim=1)
        #ic(f'ATTN SENT NORMALIZED: {x.size()}')

        # Sum attention-weighted sentences
        x = seq * x
        x = torch.sum(x, dim=1, keepdim=True)
        #ic(f'ATTN SENT SUM: {x.size()}')

        return x


class SentenceEncoder(nn.Module):
    '''
    Computes sentence-level representations
    '''

    def __init__(self, embed_dim, hidden_dim, num_layers, dropout, attn_dim, embedder):
        super(SentenceEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attn_dim = attn_dim

        self.biGRU = nn.GRU(input_size=2 * hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.word_attention = WordAttention(hidden_dim, attn_dim)
        self.word_encoder = WordEncoder(embed_dim, hidden_dim, num_layers, dropout, embedder)


    def forward(self, x : torch.Tensor, mask):
        #ic(mask.size())
        lengths = torch.sum(mask, dim=1).tolist()
        x = self.word_encoder(x, lengths)

        x = self.word_attention(x, mask)
        x, _ = self.biGRU(x)

        return x


class WordAttention(nn.Module):
    '''
    Word-level attention mechanism
    '''

    def __init__(self, hidden_dim, attn_dim):
        super(WordAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.attn_word_linear = nn.Linear(2 * hidden_dim, self.attn_dim, bias=True)
        self.tanh = nn.Tanh()
        self.attn_word_context = nn.Linear(self.attn_dim, 1, bias = False)

    def forward(self, x, mask):
        '''
        Given a sequence of word representations for a sentence, compute the attention-weighted sentence vector
        :param x: (torch.Tensor) Input. Shape: (Batch, Length, 2 * hidden_dim)
        :param mask: (torch.Tensor) Attention mask. Shape: (Batch, Length)
        :return: (torch.Tensor) Sentence vector. Shape: (Batch, 1, 2 * hidden_dim)
        '''

        seq = torch.clone(x)
        #Apply linear transformation
        x = self.tanh(self.attn_word_linear(x))
        #ic(f'ATTN WORD LINEAR: {x.size()}')
        #Compute dot product with context
        x = self.attn_word_context(x)
        #ic(f'ATTN WORD CONTEXT: {x.size()}')

        #ic(mask.size())
        #ic(mask)
        #Mask out pad values


        x[~(mask.unsqueeze(2))] = 10**(-10)
        #Compute attention weights
        x = F.softmax(x, dim=1)
        #ic(f'ATTN WORD NORMALIZED: {x.size()}')

        #ic(x)


        '''
        sum = torch.sum(x, dim=1)
        x = x / sum
        '''
        #Sum attention-weighted words
        x = seq * x
        x = torch.sum(x, dim=1, keepdim=True)
        #ic(f'ATTN WORD SUM: {x.size()}')

        return x



class WordEncoder(nn.Module):
    '''
    Computes word-level representations
    '''

    def __init__(self, embed_dim, hidden_dim, num_layers, dropout, embedder):
        super(WordEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed = embedder


        self.biGRU = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)

    def forward(self, x, lengths):
        '''
        Given input word embedding w_it from sentence s_i, outputs corresponding word-level representation

        :param x: (torch.Tensor) Input word sequence. Shape: (Batch, Length)
        :param lengths: (int[]) Length of each batch. Shape: (Batch, )
        :return: output: (torch.Tensor) Hidden state sequence for each word. Shape: (Batch, Length, 2 * hidden_dim)
        '''

        max_length = x.size(1)
        x = self.embed(x)
        device = x.device 
        
        clamped_lengths = torch.Tensor(lengths).clamp(min=1)
        packed_input = pack_padded_sequence(x, clamped_lengths, batch_first=True, enforce_sorted=False)

        #Get the sequence of hidden states corresponding to the input sequence
        packed_output, _  = self.biGRU(packed_input)

        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_length)

        lengths = torch.Tensor(lengths).to(device)
        output.masked_fill_((lengths == 0).view(-1, 1, 1), 0)

        return output




if __name__ == '__main__':
    word2vec = gensim.downloader.load('word2vec-google-news-300')
    embedder = GensimEmbedder(word2vec)
    han = HAN(2, 300, 300, 600, 1, 0.1, embedder)
    documents = [['hello my name is', 'hello world'],
                 ['what is this', 'how are you on this day today', 'okay so how about']]

    doc_tensor, attention_mask = collate_documents(word2vec, documents)
    ic(han(doc_tensor, attention_mask).size())
    '''
    #print(type(word2vec))
    sentence = 'this is sentence'.split()
    sentence = np.stack([word2vec[s] for s in sentence])
    #ic(sentence.shape)
    sentence = torch.from_numpy(sentence).unsqueeze(0)
    #ic(sentence.size())
    model = SentenceEncoder(300, 300, 1, 0.0, 500)
    model(sentence)
    '''
    #ic(model(sentence))


