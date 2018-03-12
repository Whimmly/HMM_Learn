import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import gensim
import argparse
import datetime
import numpy as np


args = argparse.ArgumentParser(
    description="Train GRU NLG model")
args.add_argument("-i", required=True, metavar="FILENAME",
                  help="filename for input file")
args.add_argument("-w2v", required=True, metavar="FILENAME",
                  help="filename for word2vec file")
args.add_argument("--iter", default=1000, metavar="FILENAME", type=int,
                  help="number of training iterations")
args = args.parse_args()


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

    def initHidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

    def initInput(self):
        return Variable(torch.zeros(1, 1, self.input_size))


class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        """
        Memory friendly iterator.
        """
        for line in open(self.fname):
            yield [word.lower() for word in line.split()]


def window_generator(model, input_size, sentence):
    input_tensor = Variable(torch.zeros(1, 1, input_size))
    for word in sentence:
        try:
            vector = model[word]
        except KeyError:
            vector = model["unk"]
        target_tensor = Variable(torch.from_numpy(vector
                                 .reshape(1, 1, -1)).float())
        yield input_tensor, target_tensor
        input_tensor = (input_tensor + target_tensor) / 2


def train(rnn, input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()
    optimizer = torch.optim.Adam(rnn.parameters())
    rnn.zero_grad()
    output, hidden = rnn(input_line_tensor, hidden)
    loss = rnn.criterion(output, target_line_tensor)

    """
    loss = 0
    for i in range(input_line_tensor.size()[0]):
        print(input_line_tensor[i].size(), hidden.size())
        output, hidden = rnn(input_line_tensor[i], hidden)
        loss += rnn.criteron(output, target_line_tensor[i])
    """
    loss.backward()
    optimizer.step()
    return output, loss


def predict(rnn, input_tensor=None):
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


def main():
    print("Start:", datetime.datetime.now())

    model = gensim.models.Word2Vec.load(args.w2v)
    input_size = model.layer1_size
    rnn = RNN(input_size, input_size * 2, 1)
    n_iters = args.iter
    print_every = n_iters / 10
    plot_every = n_iters / 10
    all_losses = []
    total_loss = 0

    for iter in range(n_iters):
        sentences = MySentences(args.i)
        for sentence in sentences:
            input_line_tensor = None
            target_line_tensor = None
            for input_tensor, target_tensor \
                    in window_generator(model, input_size, sentence):
                if input_line_tensor is None:
                    input_line_tensor = input_tensor
                input_line_tensor = torch.cat([input_line_tensor,
                                               input_tensor])
                if target_line_tensor is None:
                    target_line_tensor = target_tensor
                target_line_tensor = torch.cat([target_line_tensor,
                                               target_tensor])
            output, loss = train(rnn, input_line_tensor, target_line_tensor)
            total_loss += loss

        if iter % print_every == 0:
            print("Finished iter:", iter)

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    for word in ['april', 'country', 'month', 'food', 'references', 'desk']:
        print(word, predict_next_word(rnn, model, word), "\n")

    if len(all_losses) > 0:
        plt.figure()
        plt.plot([var.data[0] for var in all_losses])
        plt.show()

    print("End:", datetime.datetime.now())

if __name__ == "__main__":
    main()
