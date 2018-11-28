import nltk
import pickle, glob, json
from collections import Counter

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(review_files, threshold):
    """threshold: if word frequency is less than threshold, ignore word"""
    counter = Counter()
    for rf in review_files:
        f = open(rf, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            review = json.loads(line)
            text = review["reviewText"]
            tokens = nltk.tokenize.word_tokenize(text.lower())
            counter.update(tokens)

    # filter word with frequency is less than threshold
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add word to dictionary
    for word in words:
        vocab.add_word(word)

    return vocab

if __name__ == '__main__':
    data_folder = '../data/'
    threshold = 5
    review_files = glob.glob(data_folder + '*json')
    for rf in review_files:
        print(rf)
    vocab = build_vocab(review_files, threshold)

    # write vocab to file for later use
    vocab_path = '../data/vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
