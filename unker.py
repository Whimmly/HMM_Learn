from collections import Counter
import argparse

args = argparse.ArgumentParser(
    description="Replace unknown words with UNK")
args.add_argument("-v", "--vocab-size", type=int, required=True,
                  help="number of vocab words")
args.add_argument("-i", required=True, metavar="FILENAME",
                  help="filename for input file")
args.add_argument("-o", required=True, metavar="FILENAME",
                  help="filename for output file")
args = args.parse_args()


def replace_unks(sentence, vocab_set):
    split_sentence = sentence.split()
    for i in range(len(split_sentence)):
        split_sentence[i] = split_sentence[i].lower()
        if split_sentence[i] not in vocab_set:
            split_sentence[i] = "unk"
    return ' '.join(split_sentence)


def main():
    word_counter = Counter()
    with open(args.i, "r") as input_file:
        for line in input_file:
            word_counter.update([word.lower() for word in line.split()])
    vocab_set = set(word[0] for word in
                    word_counter.most_common(args.vocab_size))
    with open(args.i, "r") as input_file:
        with open(args.o, "w") as output_file:
            for line in input_file:
                output_file.write(replace_unks(line, vocab_set) + "\n")

if __name__ == "__main__":
    main()
