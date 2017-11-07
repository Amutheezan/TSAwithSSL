import time
import warnings

import numpy as np
from sklearn import preprocessing as pr, svm

from Features import *
from PreProcess import PreProcess
from SystemStore import *

warnings.filterwarnings('ignore')


class Wrapper:
    def __init__(self , label , un_label ,
                 test , iteration , train_type , test_type ,
                 confidence , confidence_diff):
        self.cons = Constants()
        self.train_contents = self.cons.TRAIN_SET.get(train_type)
        self.test_contents = self.cons.TEST_SET.get(test_type)
        self.pre_pros = PreProcess()
        self.mb = MicroBlog()
        self.ws = WritingStyle()
        self.ds = DataStore(self.cons)
        self.commons = Commons(self.cons)
        self.lexicon = Lexicon(self.pre_pros, self.cons)
        self.n_gram = NGram(self.commons ,self.cons, self.pre_pros )
        self.LABEL_LIMIT = min(self.train_contents.get(self.cons.SIZE), label)
        self.UN_LABEL_LIMIT = min(100000 , un_label)
        self.TEST_LIMIT = min(self.test_contents.get(self.cons.SIZE), test)
        self.NO_OF_ITERATIONS = max(1, iteration)
        self.POS_COUNT_LIMIT = int(self.LABEL_LIMIT * self.train_contents.get(self.cons.POS_RATIO))
        self.NEG_COUNT_LIMIT = int(self.LABEL_LIMIT * self.train_contents.get(self.cons.NEG_RATIO))
        self.NEU_COUNT_LIMIT = int(self.LABEL_LIMIT * self.train_contents.get(self.cons.NEU_RATIO))
        self.CONFIDENCE = float(confidence)
        self.CONFIDENCE_DIFF = float(confidence_diff)
        self.N_GRAM_LIMIT = 1
        self.NO_OF_MODELS = 0
        self.TRAINING_TYPE = ''
        self.final_file = ''

    def get_file_prefix(self):
        return "{0}_{1}_{2}_". \
            format(
            str(self.LABEL_LIMIT) ,
            str(self.TEST_LIMIT) , str(self.cons.DEFAULT_CLASSIFIER)
        )

    def load_training_dictionary(self):
        temp_train_dict = {}
        with open(self.train_contents.get(self.cons.TRAIN_FILE), 'r') as main_dataset:
            main = csv.reader(main_dataset)
            pos_count = 1
            neg_count = 1
            neu_count = 1
            count = 1
            for line in main:
                count += 1
                if line[ 0 ] == self.cons.NAME_POSITIVE and pos_count <= self.POS_COUNT_LIMIT:
                    temp_train_dict.update({str(count): [str(line[1]), self.cons.LABEL_POSITIVE, 1, 1]})
                    pos_count += 1

                if line[ 0 ] == self.cons.NAME_NEGATIVE and neg_count <= self.NEG_COUNT_LIMIT:
                    temp_train_dict.update({str(count): [str(line[1]), self.cons.LABEL_NEGATIVE, 1, 1]})
                    neg_count += 1

                if line[ 0 ] == self.cons.NAME_NEUTRAL and neu_count <= self.NEU_COUNT_LIMIT:
                    temp_train_dict.update({str(count): [str(line[1]), self.cons.LABEL_NEUTRAL, 1, 1]})
                    neu_count += 1

        with open(self.cons.FILE_UN_LABELED , 'r') as main_dataset:
            unlabeled = csv.reader(main_dataset)
            un_label_count = 1
            for line in unlabeled:
                count += 1
                if un_label_count <= self.UN_LABEL_LIMIT:
                    temp_train_dict.update({str(count): [str(line[0]), self.cons.UNLABELED, 0, 0]})
                    un_label_count += 1
        self.ds.TRAIN_DICT = temp_train_dict
        self.ds.POS_INITIAL = pos_count
        self.ds.NEG_INITIAL = neg_count
        self.ds.NEU_INITIAL = neu_count
        self.ds.POS_SIZE = pos_count
        self.ds.NEG_SIZE = neg_count
        self.ds.NEU_SIZE = neu_count
        return

    def save_result(self, mode = None):
        actual = []
        predicted = []
        temp_file = None
        limit = self.TEST_LIMIT
        with open(self.test_contents.get(self.cons.TEST_FILE), "r") as testFile:
            reader = csv.reader(testFile)
            count = 0
            for line in reader:
                line = list(line)
                tweet = line[1]
                s = line[0]
                s_v = self.string_to_label(s)
                actual.append(s_v)
                nl = self.predict(tweet, mode)
                predicted.append(nl)
                count = count + 1
                if count >= limit:
                    break
        result = self.commons.get_values(actual,predicted)
        pos = self.ds.POS_SIZE
        neg = self.ds.NEG_SIZE
        neu = self.ds.NEU_SIZE
        sizes = (pos,neg,neu)
        current_iteration = self.ds.CURRENT_ITERATION
        combined =  sizes + (current_iteration,) + result
        print combined
        if current_iteration > 0:
            temp_file = open(self.final_file + 'result.csv', "a+")
        if current_iteration == 0:
            temp_file = open(self.final_file + 'result.csv', "w+")
        saving_file = csv.writer(temp_file)
        if current_iteration == 0:
            saving_file.writerow(self.cons.CSV_HEADER)
        saving_file.writerow(combined)
        temp_file.close()
        return

    def string_to_label(self, s):
        if s == self.cons.NAME_POSITIVE:
            return self.cons.LABEL_POSITIVE
        if s == self.cons.NAME_NEGATIVE:
            return self.cons.LABEL_NEGATIVE
        if s == self.cons.NAME_NEUTRAL:
            return self.cons.LABEL_NEUTRAL

    def label_to_string(self,l):
        if l == self.cons.LABEL_POSITIVE:
            return self.cons.NAME_POSITIVE
        if l == self.cons.LABEL_NEGATIVE:
            return self.cons.NAME_NEGATIVE
        if l == self.cons.LABEL_NEUTRAL:
            return self.cons.NAME_NEUTRAL

    def load_matrix_sub(self , process_dict , mode , label):
        limit = self.LABEL_LIMIT
        vectors = []
        labels = []
        if limit != 0:
            keys = process_dict.keys()
            if len(keys) > 0:
                for key in keys:
                    if label == process_dict.get(key)[1]:
                        line = process_dict.get(key)[0]
                        z = self.map_tweet(line,mode)
                        vectors.append(z)
                        labels.append(label)
        return vectors , labels

    def map_tweet_n_gram_values(self , tweet):
        vector = []
        preprocessed_tweet = self.pre_pros.pre_process_tweet(tweet)
        pos_tag_tweet = self.pre_pros.pos_tag_string(preprocessed_tweet)
        for n in range(1, self.N_GRAM_LIMIT + 1 , 1):
            pos , neg , neu = self.ds._get_n_gram_(n, False)
            n_gram_score = self.n_gram.score(preprocessed_tweet , pos , neg , neu , n)
            pos , neg , neu = self.ds._get_n_gram_(n, True)
            post_n_gram_score = self.n_gram.score(pos_tag_tweet , pos , neg , neu , n)
            vector.extend(n_gram_score)
            vector.extend(post_n_gram_score)
        return vector

    def map_tweet_feature_values(self, tweet):
        vector = []
        vector.append(self.mb.emoticon_score(tweet))
        vector.append(self.mb.unicode_emoticon_score(tweet))
        vector.extend(self.ws.writing_style_vector(tweet))
        vector.extend(self.lexicon._all_in_lexicon_score_(tweet))
        return vector

    def load_iteration_dict(self, mode):
        self.ds.POS_SIZE = self.ds.POS_INITIAL
        self.ds.NEG_SIZE = self.ds.NEG_INITIAL
        self.ds.NEU_SIZE = self.ds.NEU_INITIAL
        if len(self.ds.TRAIN_DICT) > self.LABEL_LIMIT:
            for key in self.ds.TRAIN_DICT.keys():
                tweet, last_label, last_confidence, is_labeled = self.ds.TRAIN_DICT.get(key)
                if (not is_labeled) and (last_label == self.cons.UNLABELED):
                    current_label, current_confidence = self.predict_for_iteration(tweet, mode)
                    if current_label == self.cons.UNLABELED:
                        current_label = last_label
                        current_confidence = last_confidence
                    elif current_label == self.cons.LABEL_POSITIVE:
                        self.ds.POS_SIZE += 1
                    elif current_label == self.cons.LABEL_NEGATIVE:
                        self.ds.NEG_SIZE += 1
                    elif current_label == self.cons.LABEL_NEUTRAL:
                        self.ds.NEU_SIZE += 1
                    self.ds.TRAIN_DICT.update({key: [tweet, current_label, current_confidence, is_labeled]})
        return

    def generate_vectors_and_labels(self, mode):
        for n in range(1, self.N_GRAM_LIMIT + 1, 1):
            pos , pos_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_POSITIVE , n)
            neg , neg_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_NEGATIVE , n)
            neu , neu_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_NEUTRAL , n)

            self.ds._update_n_gram_(pos, neg, neu, n, False)
            self.ds._update_n_gram_(pos_p, neg_p, neu_p, n, True)

        vectors_list = []
        labels_list = []
        for label in self.cons.LABEL_TYPES:
            vec , lab = self.load_matrix_sub(self.ds.TRAIN_DICT , mode , label)
            vectors_list += vec
            labels_list += lab
        self.ds._dump_vectors_labels_(vectors_list, labels_list, mode)
        return

    def generate_model(self, mode, c_parameter, gamma):
        class_weights = self.get_class_weight
        vectors = self.ds._get_vectors_(mode)
        labels = self.ds._get_labels_(mode)
        classifier_type = self.cons.DEFAULT_CLASSIFIER
        vectors_scaled = pr.scale(np.array(vectors))
        scaler = pr.StandardScaler().fit(vectors)
        vectors_normalized = pr.normalize(vectors_scaled, norm=self.cons.L2_NORMALIZER)
        normalizer = pr.Normalizer().fit(vectors_scaled)
        vectors = vectors_normalized
        vectors = vectors.tolist()
        if classifier_type == self.cons.CLASSIFIER_SVM:
            kernel_function = self.train_contents.get(self.cons.KERNEL)
            model = svm.SVC(kernel=kernel_function , C=c_parameter ,
                            class_weight=class_weights , gamma=gamma , probability=True)
            model = model.fit(vectors, labels)
        else:
            model = None
        self.ds._dump_model_scaler_normalizer_(model, scaler, normalizer, mode)
        return

    def make_model(self, mode):
        # for mode in range(self.NO_OF_MODELS):
        c_parameter = 0.0
        gamma = 0.0
        if mode:
            c_parameter = self.train_contents.get(self.cons.C_1)
            gamma = self.train_contents.get(self.cons.GAMMA_1)
        if not mode and self.NO_OF_MODELS == 2:
            c_parameter = self.train_contents.get(self.cons.C_0)
            gamma = self.train_contents.get(self.cons.GAMMA_0)
        if not mode and self.NO_OF_MODELS == 1:
            c_parameter = self.train_contents.get(self.cons.C_SELF)
            gamma = self.train_contents.get(self.cons.GAMMA_SELF)
        self.generate_model(mode, c_parameter, gamma)

    def transform_tweet(self, tweet, mode):
        z = self.map_tweet(tweet , mode)
        try:
            z_scaled = self.ds._get_scalar_(mode).transform(z)
            z = self.ds._get_normalizer_(mode).transform([z_scaled])
        except ValueError:
            z_scaled = self.ds._get_scalar_(mode).transform([z])
            z = self.ds._get_normalizer_(mode).transform(z_scaled)
        z = z[0].tolist()
        return z

    @property
    def get_class_weight(self):
        pos = self.ds.POS_SIZE
        neg = self.ds.NEG_SIZE
        neu = self.ds.NEU_SIZE
        is_pos_max = False
        is_neg_max = False
        is_neu_max = False
        maximum = max(pos, neg, neu)
        if maximum == pos:
            is_pos_max = True
        if maximum == neg:
            is_neg_max = True
        if maximum == neu:
            is_neu_max = True
        weights = dict()
        if is_pos_max:
            weights[self.cons.LABEL_POSITIVE] = 1.0
        else:
            weights[self.cons.LABEL_POSITIVE] = (1.0 * maximum) / pos
        if is_neg_max:
            weights[self.cons.LABEL_NEGATIVE] = 1.0
        else:
            weights[self.cons.LABEL_NEGATIVE] = (1.0 * maximum) / neg
        if is_neu_max:
            weights[self.cons.LABEL_NEUTRAL] = 1.0
        else:
            weights[self.cons.LABEL_NEUTRAL] = (1.0 * maximum) / neu
        return weights

    def common_run(self, mode):
        self.generate_vectors_and_labels(mode)
        self.make_model(mode)
        self.save_result(mode)

    def do_training(self):

        self.ds.CURRENT_ITERATION = 0
        time_list = [time.time()]
        self.load_training_dictionary()
        self.common_run(0)

        while self.ds.CURRENT_ITERATION < 10:

            self.load_iteration_dict(1)
            self.ds.CURRENT_ITERATION += 1
            self.common_run(1)
            self.ds.CURRENT_ITERATION -= 1

            self.load_iteration_dict(0)
            self.ds.CURRENT_ITERATION += 1
            self.common_run(0)
          

        time_list.append(time.time())
        print self.commons.temp_difference_cal(time_list)

    def predict(self, tweet, mode = None):
        pass

    def predict_for_iteration(self, tweet, mode = None):
        pass

    def map_tweet(self , line , mode):
        pass
