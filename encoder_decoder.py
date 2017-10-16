import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import unicodedata
import re
import os
import pickle
from torch.autograd import Variable
import torch.nn.functional as F


MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
use_cuda = torch.cuda.is_available()
hidden_size = 256

# lang_translate_path = '/home/knswamy153/FlaskApp/flask_rest_service/lang_translate/lang_translate.pkl'
# pkl_file = open(lang_translate_path, 'rb')
# input_output_lang = pickle.load(pkl_file)
# input_lang = input_output_lang['input_lang']
# output_lang = input_output_lang['output_lang']

# pkl_file.close()



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def load_dict(self,lang_data):
        self.name = lang_data['name']
        self.word2index = lang_data['word2index']
        self.word2count = lang_data['word2count']
        self.index2word = lang_data['index2word']
        self.n_words = lang_data['n_words'] 

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result 

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result   

def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
        
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def main(args):
    encoder_path = '/home/knswamy153/FlaskApp/flask_rest_service/lang_translate'
    decoder_path = '/home/knswamy153/FlaskApp/flask_rest_service/lang_translate'
    lang_data_path = '/home/knswamy153/FlaskApp/flask_rest_service/lang_translate'

    if (args.which_lang == 'french'):
        input_lang_data_path = os.path.join(lang_data_path, 'input_lang_data.pkl')
        output_lang_data_path = os.path.join(lang_data_path, 'output_lang_data.pkl')
        encoder_lang_path = os.path.join(lang_data_path, 'encoder1.pth')
        decoder_lang_path = os.path.join(lang_data_path, 'attn_decoder1.pth')

    else:
        input_lang_data_path = os.path.join(lang_data_path, 'input_lang_ger_data.pkl')
        output_lang_data_path = os.path.join(lang_data_path, 'output_lang_ger_data.pkl')
        encoder_lang_path = os.path.join(lang_data_path, 'ger_encoder1.pth')
        decoder_lang_path = os.path.join(lang_data_path, 'ger_attn_decoder1.pth')

    pkl_file = open(input_lang_data_path, 'rb')
    input_lang_data = pickle.load(pkl_file)
    input_lang = Lang('eng')
    input_lang.load_dict(input_lang_data)
    
    pkl_file = open(output_lang_data_path, 'rb')
    output_lang_data = pickle.load(pkl_file)
    output_lang = Lang('fra')
    output_lang.load_dict(output_lang_data)

    pkl_file.close()

    #print input_lang.n_words
    encoder1 = EncoderRNN(input_lang.n_words,hidden_size)
    encoder1.load_state_dict(torch.load(encoder_lang_path))

    decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, 1, dropout_p=0.1)
    decoder1.load_state_dict(torch.load(decoder_lang_path))

    # Add a period to end of sentence if not present and then normalize sentence
    norm_sentence = args.input_sentence
    if norm_sentence[-1:] != ".":
        norm_sentence = norm_sentence + "."
    norm_sentence = normalizeString(norm_sentence)
    try:
        output_words, attentions = evaluate(encoder1, decoder1, input_lang, output_lang, norm_sentence)  
        output_sentence = ' '.join(output_words)
        del encoder1, decoder1
        return output_sentence
    except KeyError as err:
        del encoder1, decoder1
        return "Key error:"+str(err)    

if __name__ == "__main__":
    lang_arg_parser = argparse.ArgumentParser(description="parser for language")
    
    lang_arg_parser.add_argument("--which-lang", type=str, required=True,
                                 help="To which language you want to translate")
    lang_arg_parser.add_argument("--input-sentence", type=unicode, required=True,
                                 help="Input Language sentence")
    args = lang_arg_parser.parse_args()

    #eng_lang_text = u"I am running."
    #return eng_lang_text
    #which_lang = "french"
    output_sentence = main(args)
    #return {"output_sentence": output_sentence}
    print output_sentence                            