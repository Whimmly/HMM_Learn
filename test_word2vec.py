import gensim, logging, argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

args = argparse.ArgumentParser(
    description="Train word2vec model")
args.add_argument("-i", required=True, metavar="FILENAME",
                  help="filename for input file")
args = args.parse_args() 

if __name__ == "__main__":
    model = gensim.models.Word2Vec.load(args.i)
    for word in ['april', 'country', 'month', 'food', 'references', 'desk']:
        print(word, model.most_similar(word))
