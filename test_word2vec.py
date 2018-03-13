#!/usr/bin/env python
import gensim, logging, argparse
import numpy as np

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.INFO)

args = argparse.ArgumentParser(
    description="Train word2vec model")
args.add_argument("-i", required=True, metavar="FILENAME",
                  help="filename for input file")
args = args.parse_args()

def likeliest_next_word(model, sentence):
    max_score = -100
    max_word = None
    all_sentences = [sentence + [word] for word in model.wv.vocab]
    all_scores = model.score(all_sentences, len(all_sentences))
    for i, curr_score in enumerate(all_scores):
        if curr_score > max_score:
            max_score = curr_score
            max_word = all_sentences[i][-1]
    return max_word

if __name__ == "__main__":
    model = gensim.models.Word2Vec.load(args.i)
    for word in ['april', 'country', 'month', 'food', 'references', 'desk']:
        print(word, likeliest_next_word(model, word.split()))
