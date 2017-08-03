import numpy as np
from collections import Counter
import re
import scipy.io as sio

def extract_data (data_file_path):
    # data = sio.loadmat('data/mpqa/bigram_mpqa.mat')
    data_samples = []
    with open(data_file_path, "r") as file_:          #making vector of words from the given file
        x = ([re.compile("[a-zA-Z\']*[a-zA-Z]").findall(review) for review in file_])
    with open(data_file_path, "r") as file_:
        y = ([re.compile("^[01]").findall(review) for review in file_])
    return x,y

class MNB_classifier():
    def __init__ (self, file_path):
        (self.samples, self.lables) = extract_data(file_path)
        self.words_dic = {}
        self.pos_data = []
        self.neg_data = []
        self.words_in_pos_data = Counter()
        self.words_in_neg_data = Counter()
        self.index = 0
        self.being_pos_prob = None
        self.words_in_poses_prob = {}
        self.words_in_neges_prob = {}
        self.words_appearance_prob = {}

    def build_words_list (self):
        for i in xrange(len(self.samples)):
            if self.lables[i] == 1:
                self.pos_data.append(self.samples[i])
            else:
                self.neg_data.append(self.samples[i])
            print (len(self.pos_data))
            print (len(self.neg_data))
        # self.pos_data = extract_data(self.pos_file_path)
        # self.neg_data = extract_data(self.neg_file_path)
        for pos_review, neg_review in zip(self.pos_data, self.neg_data):
            for pos_word, neg_word in zip(pos_review, neg_review):
                if not pos_word in self.words_dic:
                    self.words_dic[pos_word] = self.index
                    self.index += 1
                if not neg_word in self.words_dic:
                    self.words_dic[neg_word] = self.index
                    self.index += 1

    def build_samples_feature_vectors (self):
        for pos_review, neg_review in zip(self.pos_data, self.neg_data):
            for word1, word2 in zip(pos_review, neg_review):
                self.words_in_pos_data.update(word1)
                self.words_in_neg_data.update(word2)
        for word in self.words_dic:                  #For avoiding probability = 0
            self.words_in_pos_data.update([word])
            self.words_in_neg_data.update([word])

    def compute_parameters (self):
        (num_of_pos_words, num_of_neg_words) = (0,0)
        for word1, word2 in zip(self.words_in_pos_data, self.words_in_neg_data):
            num_of_pos_words += self.words_in_pos_data[word1]
            num_of_neg_words += self.words_in_neg_data[word2]
        self.words_appearance_prob = {word:(self.words_in_pos_data[word] + self.words_in_neg_data[word]) * 1.0/
                (num_of_pos_words + num_of_neg_words) for word in self.words_dic}
        self.being_pos_prob = len(self.pos_data)*1.0 / (len(self.pos_data) + len(self.neg_data))
        self.words_in_poses_prob = {word:self.words_in_pos_data[word]*1.0/num_of_pos_words for word in self.words_dic}
        self.words_in_neges_prob = {word:self.words_in_neg_data[word]*1.0/num_of_neg_words for word in self.words_dic}

    def print_predicted_result (self, data_file):
        counter = 0
        (data_vec, data_lable) = extract_data(data_file)
        for review, i in zip(data_vec, xrange(len(data_vec))):
            try:
                prob = self.predict(review)
                if (data_lable[i] == 1 and prob > 0.5) or (data_lable[i] == 0 and prob <= 0.5):
                    counter += 1
            except (ZeroDivisionError) as e:
                pass
        print (counter*1.0 / len(data_vec))

    def predict (self, review):
        probability = 1.0
        for word in review:
            if word in self.words_dic:
                probability = probability * self.words_in_poses_prob[word] / self.words_appearance_prob[word]
        return probability * self.being_pos_prob

if __name__ == "__main__":
    MNB_C = MNB_classifier("./data/mpqa/mpqa_t4.dat")
    MNB_C.build_words_list()
    MNB_C.build_samples_feature_vectors()
    MNB_C.compute_parameters()
    MNB_C.print_predicted_result("./data/mpqa/mpqa_t4.dat")
