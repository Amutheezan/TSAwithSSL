import csv
import time
import warnings

import numpy as np
from sklearn import preprocessing as pr , svm

from SystemStore import *
from Features import *
from PreProcess import PreProcess

warnings.filterwarnings('ignore')


class Wrapper:
    def __init__(self , label , un_label , test, iteration):
        self.cons = Constants()
        self.pre_pros = PreProcess()
        self.mb = MicroBlog()
        self.ws = WritingStyle()
        self.ds = DataStore(self.cons)
        self.commons = Commons(self.cons)
        self.lexicon = Lexicon(self.pre_pros, self.cons)
        self.n_gram = NGram(self.commons ,self.cons, self.pre_pros )
        self.LABEL_LIMIT = min(self.cons.LABEL_DATA_SET_SIZE , label)
        self.UN_LABEL_LIMIT = min(100000 , un_label)
        self.TEST_LIMIT = min(self.cons.TEST_DATA_SET_SIZE , test)
        self.NO_OF_ITERATIONS = max(1, iteration)
        self.POS_COUNT_LIMIT = int(self.LABEL_LIMIT * self.cons.POS_RATIO)
        self.NEG_COUNT_LIMIT = int(self.LABEL_LIMIT * self.cons.NEG_RATIO)
        self.NEU_COUNT_LIMIT = int(self.LABEL_LIMIT * self.cons.NEU_RATIO)
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
        with open(self.cons.FILE_LABELED , 'r') as main_dataset:
            main = csv.reader(main_dataset)
            pos_count = 1
            neg_count = 1
            neu_count = 1
            count = 1
            for line in main:
                count += 1
                if line[ 0 ] == self.cons.NAME_POSITIVE and pos_count <= self.POS_COUNT_LIMIT:
                    temp_train_dict.update({str(count): [str(line[ 1 ]), self.cons.LABEL_POSITIVE, 1, 1, self.cons.NO_TOPIC, 0]})
                    pos_count += 1

                if line[ 0 ] == self.cons.NAME_NEGATIVE and neg_count <= self.NEG_COUNT_LIMIT:
                    temp_train_dict.update({str(count): [str(line[ 1 ]),self.cons.LABEL_NEGATIVE, 1, 1, self.cons.NO_TOPIC, 0]})
                    neg_count += 1

                if line[ 0 ] == self.cons.NAME_NEUTRAL and neu_count <= self.NEU_COUNT_LIMIT:
                    temp_train_dict.update({str(count): [str(line[ 1 ]),self.cons.LABEL_NEUTRAL, 1, 1, self.cons.NO_TOPIC, 0]})
                    neu_count += 1

        with open(self.cons.FILE_UN_LABELED , 'r') as main_dataset:
            unlabeled = csv.reader(main_dataset)
            un_label_count = 1
            for line in unlabeled:
                count +=1
                if un_label_count <= self.UN_LABEL_LIMIT:
                    temp_train_dict.update({str(count): [ str(line[ 0 ]) , self.cons.UNLABELED, 0, 0, self.cons.NO_TOPIC, 0 ]})
                    un_label_count += 1
        self.ds.TRAIN_DICT = temp_train_dict
        self.ds.POS_INITIAL = pos_count
        self.ds.NEG_INITIAL = neg_count
        self.ds.NEU_INITIAL = neu_count
        self.ds.POS_SIZE = pos_count
        self.ds.NEG_SIZE = neg_count
        self.ds.NEU_SIZE = neu_count
        return
    
    def save_result(self, test_type):
        actual = []
        predicted = []
        temp_file = None
        limit = self.TEST_LIMIT
        with open(self.cons.FILE_TEST, "r") as testFile:
            reader = csv.reader(testFile)
            count = 0
            for line in reader:
                line = list(line)
                if line[0] == test_type:
                    tweet = line[2]
                    s = line[1]
                    s_v = self.string_to_label(s)
                    actual.append(s_v)
                    nl = self.predict(tweet)
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
        combined = (test_type,) + sizes + (current_iteration,) + result
        print combined
        if current_iteration > 0:
            temp_file = open(self.final_file + test_type +'result.csv',"a+")
        if current_iteration == 0:
            temp_file = open(self.final_file + test_type +'result.csv',"w+")
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
        if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
            vectors = {}
            labels = {}
        else:
            vectors = []
            labels = []
        if limit != 0:
            keys = process_dict.keys()
            if len(keys) > 0:
                for key in keys:
                    if label == process_dict.get(key)[1]:
                        line = process_dict.get(key)[0]
                        z = self.map_tweet(line,mode)
                        if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
                            vectors.update({key : z})
                            labels.update({key : label})
                        else:
                            vectors.append(z)
                            labels.append(label)
        return vectors , labels

    def map_tweet_n_gram_values(self , tweet):
        vector = []

        preprocessed_tweet = self.pre_pros.pre_process_tweet(tweet)
        pos_tag_tweet = self.pre_pros.pos_tag_string(preprocessed_tweet)

        uni_gram_score = self.n_gram.score(preprocessed_tweet , self.ds.POS_UNI_GRAM , self.ds.NEG_UNI_GRAM ,
                                           self.ds.NEU_UNI_GRAM , 1)
        post_uni_gram_score = self.n_gram.score(pos_tag_tweet , self.ds.POS_POST_UNI_GRAM ,
                                                self.ds.NEG_POST_UNI_GRAM ,
                                                self.ds.NEU_POST_UNI_GRAM , 1)
        vector.append(self.mb.emoticon_score(tweet))
        vector.append(self.mb.unicode_emoticon_score(tweet))
        vector.extend(self.ws.writing_style_vector(tweet))
        vector.extend(uni_gram_score)
        vector.extend(post_uni_gram_score)

        return vector

    def map_tweet_feature_values(self, tweet):
        vector = self.lexicon._all_in_lexicon_score_(tweet)
        return vector

    def load_iteration_dict(self):
        self.ds.POS_SIZE = self.ds.POS_INITIAL
        self.ds.NEG_SIZE = self.ds.NEG_INITIAL
        self.ds.NEU_SIZE = self.ds.NEU_INITIAL
        if len(self.ds.TRAIN_DICT) > self.LABEL_LIMIT:
            for key in self.ds.TRAIN_DICT.keys():
                tweet , last_label, last_confidence, is_labeled, last_topic, last_value = self.ds.TRAIN_DICT.get(key)
                if not is_labeled:
                    current_label,current_confidence = self.predict_for_iteration(tweet , last_label)
                    if current_label == self.cons.UNLABELED:
                        current_label = last_label
                        current_confidence = last_confidence
                    elif current_label == self.cons.LABEL_POSITIVE:
                        self.ds.POS_SIZE += 1
                    elif current_label == self.cons.LABEL_NEGATIVE:
                        self.ds.NEG_SIZE += 1
                    elif current_label == self.cons.LABEL_NEUTRAL:
                        self.ds.NEU_SIZE += 1
                    self.ds.TRAIN_DICT.update({key: [tweet , current_label, current_confidence, is_labeled, last_topic, last_value ]})
        self.ds._increment_iteration_()
        return

    def generate_vectors_and_labels(self):
        pos , pos_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_POSITIVE , 1)
        neg , neg_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_NEGATIVE , 1)
        neu , neu_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_NEUTRAL , 1)

        self.ds._update_uni_gram_(pos , neg , neu , False)
        self.ds._update_uni_gram_(pos_p , neg_p , neu_p , True)
        for mode in range(self.NO_OF_MODELS):
            if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
                vectors_dict = {}
                labels_dict = {}
            else:
                vectors_list = []
                labels_list = []
            for label in self.cons.LABEL_TYPES:
                vec , lab = self.load_matrix_sub(self.ds.TRAIN_DICT , mode , label)
                if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
                    vectors_dict.update(vec)
                    labels_dict.update(lab)
                else:
                    vectors_list += vec
                    labels_list += lab
            if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
                self.ds._dump_vectors_labels_(vectors_dict , labels_dict , mode)
            else:
                self.ds._dump_vectors_labels_(vectors_list , labels_list , mode)
        return

    def get_vectors_and_labels(self,mode, topic):
        full_vectors = self.ds._get_vectors_(mode)
        full_labels = self.ds._get_labels_(mode)
        if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
            selected_vectors = []
            selected_labels = []
            for key in full_vectors.keys():
                if self.ds.TRAIN_DICT.get(key)[4] == topic:
                    selected_vectors.append(full_vectors.get(key))
                    selected_labels.append(full_labels.get(key))
                elif topic == self.cons.NO_TOPIC:
                    selected_vectors.append(full_vectors.get(key))
                    selected_labels.append(full_labels.get(key))
            return selected_vectors, selected_labels
        else:
            return full_vectors, full_labels

    def generate_model(self , mode , c_parameter , gamma , topic):
        class_weights = self.get_class_weight()
        vectors,labels = self.get_vectors_and_labels(mode, topic)
        classifier_type = self.cons.DEFAULT_CLASSIFIER
        vectors_scaled = pr.scale(np.array(vectors))
        scaler = pr.StandardScaler().fit(vectors)
        vectors_normalized = pr.normalize(vectors_scaled , norm='l2')
        normalizer = pr.Normalizer().fit(vectors_scaled)
        vectors = vectors_normalized
        vectors = vectors.tolist()
        if classifier_type == self.cons.CLASSIFIER_SVM:
            kernel_function = self.cons.DEFAULT_KERNEL
            model = svm.SVC(kernel=kernel_function , C=c_parameter ,
                            class_weight=class_weights , gamma=gamma , probability=True)
            model.fit(vectors, labels)
        else:
            model = None
        self.ds._dump_model_scaler_normalizer_(model , scaler , normalizer , mode, topic)
        return

    def make_model(self , topic):
        for mode in range(self.NO_OF_MODELS):
            c_parameter = 0.0
            gamma = 0.0
            if mode:
                c_parameter = self.cons.DEFAULT_C_PARAMETER
                gamma = self.cons.DEFAULT_GAMMA_SVM
            if not mode and self.NO_OF_MODELS == 2:
                c_parameter = self.cons.DEFAULT_C_PARAMETER_0
                gamma = self.cons.DEFAULT_GAMMA_SVM_0
            if not mode and self.NO_OF_MODELS == 1:
                c_parameter = self.cons.DEFAULT_C_PARAMETER_SELF
                gamma = self.cons.DEFAULT_GAMMA_SVM_SELF
            self.generate_model(mode , c_parameter , gamma , topic)

    def transform_tweet(self , tweet , mode, topic):
        z = self.map_tweet(tweet , mode)
        z_scaled = self.ds._get_scalar_(mode, topic).transform(z)
        z = self.ds._get_normalizer_(mode,topic).transform([ z_scaled ])
        z = z[0].tolist()
        return z

    def common_run(self,test_type):
        self.generate_vectors_and_labels()
        self.make_model(self.cons.NO_TOPIC)
        self.save_result(test_type)

    def get_class_weight(self):
        pos = self.ds.POS_SIZE
        neg = self.ds.NEG_SIZE
        neu = self.ds.NEU_SIZE
        weights = dict()
        weights[self.cons.LABEL_POSITIVE] = (1.0 * neu) / pos
        weights[self.cons.LABEL_NEGATIVE] = (1.0 * neu) / neg
        weights[self.cons.LABEL_NEUTRAL] = 1.0
        return weights

    def do_training(self):
        for test_type in self.cons.TEST_TYPES:
            self.ds.CURRENT_ITERATION = 0
            time_list = [time.time()]
            self.load_training_dictionary()
            self.common_run(test_type)
            while self.ds.CURRENT_ITERATION < 10:
                self.load_iteration_dict()
                self.common_run(test_type)

            time_list.append(time.time())
            print self.commons.temp_difference_cal(time_list)

    def predict(self , tweet):
        pass

    def predict_for_iteration(self , tweet , last_label):
        pass

    def map_tweet(self , line , mode):
        pass
