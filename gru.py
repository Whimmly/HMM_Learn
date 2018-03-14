#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import gensim
import argparse
import datetime
import random
import numpy as np
from pprint import pprint

args = argparse.ArgumentParser(
    description="Train GRU NLG model")
args.add_argument("-i", default=None, metavar="FILENAME",
                  help="filename for input model")
args.add_argument("-o", default="checkpoints/gru.model", metavar="FILENAME",
                  help="filename for output model")
args.add_argument("--data", "-d", required=True, metavar="FILENAME",
                  help="filename for input data")
args.add_argument("--w2v", required=True, metavar="FILENAME",
                  help="filename for word2vec file")
args.add_argument("--epochs", default=1000, type=int,
                  help="number of training iterations")
args.add_argument("--batch_size", default=100, type=int,
                  help="number of training examples per batch")
args.add_argument("--window_size", default=5, type=int,
                  help="number of training examples per batch")
args = args.parse_args()

PADDING_TOK = "PADDING_TOK"


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.encoder = torch.nn.GRU(input_size, hidden_size, num_layers)
        self.decoder = torch.nn.Linear(hidden_size, input_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.criterion = nn.MSELoss()

    def forward(self, input_tensor, hidden):
        output, hidden = self.encoder(input_tensor, hidden)
        return self.decoder(output), hidden

    def initHidden(self, batch_size=1):
        return Variable(torch.zeros(1, batch_size, self.hidden_size))

    def initInput(self, batch_size=1):
        return Variable(torch.zeros(1, batch_size, self.input_size))


class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        """
        Memory friendly iterator.
        """
        for line in open(self.fname):
            yield [word.lower() for word in line.split()]


def train(rnn, input_line_tensor, target_line_tensor):
    """
    Train the RNN on one batch.
    """
    assert(input_line_tensor.size()[0] == target_line_tensor.size()[0])
    assert(input_line_tensor.size()[1:] == (args.window_size, rnn.input_size))
    assert(target_line_tensor.size()[1:] == (1, rnn.input_size))

    # Transpose tensors to fit pytorch specs(seq_len, batch_size, input_size)
    input_line_tensor = input_line_tensor.permute(1, 0, 2)
    target_line_tensor = target_line_tensor.permute(1, 0, 2)

    hidden = rnn.initHidden(input_line_tensor.size()[1])
    optimizer = torch.optim.Adam(rnn.parameters())
    rnn.zero_grad()  # Equivalent to optimizer.zero_grad()
    output, hidden = rnn(input_line_tensor, hidden)

    loss = rnn.criterion(output[-1:, :, :], target_line_tensor)
    loss.backward()
    optimizer.step()
    return output, loss


def predict(rnn, input_tensor=None):
    """
    Predict the next output vector from the input.
    """
    if input_tensor is None:
        input_tensor = rnn.initInput()
    elif type(input_tensor) is np.ndarray:
        input_tensor = Variable(torch.from_numpy(input_tensor
                                .reshape(1, 1, -1)).float())
    hidden = rnn.initHidden()
    output, hidden = rnn(input_tensor, hidden)
    return output


def predict_next_word(rnn, model, word):
    """
    Predict next possible words from a single word.
    """
    prediction = predict(rnn, model[word])
    return model.most_similar(prediction[0].data.numpy())


def window_generator(model, input_size, sentence,
                     window_size=args.window_size):
    """
    Create window input and target tensors.

    DEPRECATED
    """
    input_tensor = torch.zeros(window_size, 1, input_size)
    for word in sentence:
        try:
            vector = model[word]
        except KeyError:
            vector = model["unk"]
        target_tensor = Variable(torch.from_numpy(vector
                                 .reshape(1, 1, -1)).float())
        yield input_tensor, target_tensor
        input_tensor = torch.cat([input_tensor[1:], target_tensor])


def word_window_generator(sentence, window_size=args.window_size):
    """
    Arrange sentence into a generator of window, word pairs.
    """
    input_window = [PADDING_TOK] * window_size
    for word in sentence:
        yield input_window, [word]
        input_window = input_window[1:] + [word]


def strings_to_vectors(word2vec, batch):
    """
    Convert the strings to vectors, and arrange them in proper 3D matrix.
    """
    batch_vectors = []
    for seq in batch:
        seq_vectors = []
        for word in seq:
            if word == PADDING_TOK:
                vector = np.zeros(word2vec.layer1_size)
            else:
                try:
                    vector = word2vec[word]
                except KeyError:
                    vector = word2vec["unk"]
            seq_vectors.append(vector)
        batch_vectors.append(seq_vectors)
    return np.array(batch_vectors)


def strings_to_tensors(word2vec, data_batch):
    """
    Convert string array to word vector tensor array.

    Args:
    - data_batch: string python list with dimension:
                  (batch_size, window_length, input_size)
    """
    input_batch = [datapoint[0] for datapoint in data_batch]
    target_batch = [datapoint[1] for datapoint in data_batch]
    input_vector_batch = strings_to_vectors(word2vec, input_batch)
    target_vector_batch = strings_to_vectors(word2vec, target_batch)

    input_tensor = torch.autograd.Variable(
        torch.from_numpy(input_vector_batch).float())
    target_tensor = torch.autograd.Variable(
        torch.from_numpy(target_vector_batch).float())

    return input_tensor, target_tensor


def main():
    print("Start:", datetime.datetime.now())

    word2vec = gensim.models.Word2Vec.load(args.w2v)
    input_size = word2vec.layer1_size
    rnn = RNN(input_size, input_size * 2, 1)
    if args.i is not None:
        rnn.load_state_dict(torch.load(args.i))
    n_epochs = args.epochs
    batch_size = args.batch_size
    print_every = 100 * batch_size
    plot_every = 100 * batch_size
    save_every = 1
    all_losses = []
    total_loss = 0

    for epoch in range(n_epochs):
        """
        Load all words into memory
        Create windows
        Shuffle dataset
        Batch train
        """
        sentences = MySentences(args.data)
        datapoints = [datapoint
                      for sentence in sentences
                      for datapoint in word_window_generator(sentence)]
        print("Loaded sentences")

        random.shuffle(datapoints)
        print("Shuffled data")

        for i in range(0, len(datapoints), batch_size):
            input_line_tensor, target_line_tensor = strings_to_tensors(
                word2vec, datapoints[i:i + batch_size])
            output, loss = train(rnn, input_line_tensor, target_line_tensor)
            total_loss += loss

            if i % print_every == 0 and i != 0:
                print("Finished batch:", i // batch_size)

            if i % plot_every == 0 and i != 0:
                batch_loss = total_loss.data[0] / min(batch_size,
                                                      len(datapoints) - i)
                print("Loss at batch", i // batch_size, ":", batch_loss)
                all_losses.append(batch_loss)
                total_loss = 0



        print("Finished epoch:", epoch)
        if epoch % save_every == 0:
            data_fname = args.data.split('/')[-1]
            torch.save(rnn.state_dict(), args.o + "." + data_fname + "." + 
                       str(batch_size) + "." + str(epoch))
            print("Saved checkpoint")

    for word in ['april', 'country', 'month', 'food', 'references', 'desk',
                 'is']:
        print("\n", word, ":")
        pprint(predict_next_word(rnn, word2vec, word))

    # Visualize loss over time
    if len(all_losses) > 0:
        print("All losses: ")
        pprint(all_losses)
        try:
            plt.figure()
            plt.plot(all_losses)
            plt.show()
        except Exception as e:
            print("Unable to show graph due to error:", e)

    print("End:", datetime.datetime.now())

if __name__ == "__main__":
    main()
