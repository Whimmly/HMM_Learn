import gensim, logging, argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

args = argparse.ArgumentParser(
    description="Train word2vec model")
args.add_argument("-i", required=True, metavar="FILENAME",
                  help="filename for input file")
args.add_argument("-o", required=True, metavar="FILENAME",
                  help="filename for output file")
args = args.parse_args() 

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open(self.fname):
            yield [word.lower() for word in line.split()]

if __name__ == "__main__":
    sentences = MySentences(args.i)
    model = gensim.models.Word2Vec(sentences, min_count=5, iter=5, size=100)
    model.save(args.o)

