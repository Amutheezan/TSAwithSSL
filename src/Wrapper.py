import csv
import warnings
import time
from Configuration import Configuration
from Commons import Commons
from DataStore import DataStore
from Features import MicroBlog , Lexicon , WritingStyle , NGram
from PreProcess import PreProcess
warnings.filterwarnings('ignore')


class Wrapper:
    def __init__(self , label , un_label , test):
        self.ds = DataStore()
        self.config = Configuration()
        self.pre_pros = PreProcess()
        self.commons = Commons(self.config)
        self.mb = MicroBlog()
        self.lexicon = Lexicon(self.pre_pros)
        self.ws = WritingStyle()
        self.n_gram = NGram(self.commons , self.pre_pros)
        self.LABEL_LIMIT = min(self.config.LABEL_DATA_SET_SIZE , label)
        self.UN_LABEL_LIMIT = min(100000 , un_label)
        self.TEST_LIMIT = min(self.config.TEST_DATA_SET_SIZE , test)
        self.POS_COUNT_LIMIT = int(self.LABEL_LIMIT * self.config.POS_RATIO)
        self.NEG_COUNT_LIMIT = int(self.LABEL_LIMIT * self.config.NEG_RATIO)
        self.NEU_COUNT_LIMIT = int(self.LABEL_LIMIT * self.config.NEU_RATIO)
        self.final_file = ''

    def get_file_prefix(self):
        return "{0}_{1}_{2}_". \
            format(
            str(self.LABEL_LIMIT) ,
            str(self.TEST_LIMIT) , str(self.config.DEFAULT_CLASSIFIER)
        )

    def load_initial_dictionaries(self):
        pos_dict = {}
        neg_dict = {}
        neu_dict = {}
        un_label_dict = {}
        with open(self.config.FILE_LABELED , 'r') as main_dataset:
            main = csv.reader(main_dataset)
            pos_count = 1
            neg_count = 1
            neu_count = 1
            for line in main:
                if line[ 0 ] == self.config.NAME_POSITIVE and pos_count <= self.POS_COUNT_LIMIT:
                    pos_dict.update({str(pos_count): str(line[ 1 ])})
                    pos_count += 1

                if line[ 0 ] == self.config.NAME_NEGATIVE and neg_count <= self.NEG_COUNT_LIMIT:
                    neg_dict.update({str(neg_count): str(line[ 1 ])})
                    neg_count += 1

                if line[ 0 ] == self.config.NAME_NEUTRAL and neu_count <= self.NEU_COUNT_LIMIT:
                    neu_dict.update({str(neu_count): str(line[ 1 ])})
                    neu_count += 1

        with open(self.config.FILE_UN_LABELED , 'r') as main_dataset:
            unlabeled = csv.reader(main_dataset)
            un_label_count = 1
            for line in unlabeled:
                if un_label_count <= self.UN_LABEL_LIMIT:
                    un_label_dict.update({str(un_label_count): [ str(line[ 0 ]) , self.config.UNLABELED ]})
                    un_label_count += 1

        return pos_dict , neg_dict , neu_dict , un_label_dict
    
    def save_result(self, is_iteration, test_type):
        actual = []
        predicted = []
        temp_file = None
        limit = self.TEST_LIMIT
        with open(self.config.FILE_TEST, "r") as testFile:
            reader = csv.reader(testFile)
            count = 0
            for line in reader:
                line = list(line)
                if line[0] == test_type:
                    tweet = line[2]
                    s = line[1]
                    s_v = self.string_to_label(s)
                    actual.append(s_v)
                    nl = self.predict(tweet, is_iteration)
                    predicted.append(nl)
                    count = count + 1
                    if count >= limit:
                        break
        result = self.commons.get_values(actual,predicted)
        sizes = self.get_size(is_iteration)
        current_iteration = self.ds.CURRENT_ITERATION
        combined = (test_type,) + sizes + (current_iteration,) + result
        print combined
        if is_iteration:
            temp_file = open(self.final_file + test_type +'result.csv',"a+")
        if not is_iteration:
            temp_file = open(self.final_file + test_type +'result.csv',"w+")
        saving_file = csv.writer(temp_file)
        if not is_iteration:
            saving_file.writerow(self.config.CSV_HEADER)
        saving_file.writerow(combined)
        self.ds.CURRENT_ITERATION += 1
        temp_file.close()
        return

    def string_to_label(self, s):
        if s == self.config.NAME_POSITIVE:
            return self.config.LABEL_POSITIVE
        if s == self.config.NAME_NEGATIVE:
            return self.config.LABEL_NEGATIVE
        if s == self.config.NAME_NEUTRAL:
            return self.config.LABEL_NEUTRAL

    def label_to_string(self,l):
        if l == self.config.LABEL_POSITIVE:
            return self.config.NAME_POSITIVE
        if l == self.config.LABEL_NEGATIVE:
            return self.config.NAME_NEGATIVE
        if l == self.config.LABEL_NEUTRAL:
            return self.config.NAME_NEUTRAL

    def load_matrix_sub(self , process_dict , mode , label , is_iteration):
        """
        :param process_dict:
        :param mode:
        :param label:
        :param is_iteration:
        :return:
        """
        limit = self.LABEL_LIMIT
        if limit != 0:
            keys = process_dict.keys()
            if len(keys) > 0:
                vectors = []
                labels = []
                for key in keys:
                    line = process_dict.get(key)
                    z = self.map_tweet(line , mode , is_iteration)
                    vectors.append(z)
                    labels.append(float(label))
            else:
                vectors = [ ]
                labels = [ ]
        else:
            vectors = [ ]
            labels = [ ]
        return vectors , labels

    def map_tweet_n_gram_values(self , tweet , is_iteration):
        """
        :param tweet: 
        :param is_iteration: 
        :return: 
        """
        vector = [ ]

        preprocessed_tweet = self.pre_pros.pre_process_tweet(tweet)
        pos_tag_tweet = self.pre_pros.pos_tag_string(preprocessed_tweet)

        if not is_iteration:
            uni_gram_score = self.n_gram.score(preprocessed_tweet , self.ds.POS_UNI_GRAM , self.ds.NEG_UNI_GRAM ,
                                               self.ds.NEU_UNI_GRAM , 1)
            post_uni_gram_score = self.n_gram.score(pos_tag_tweet , self.ds.POS_POST_UNI_GRAM ,
                                                    self.ds.NEG_POST_UNI_GRAM ,
                                                    self.ds.NEU_POST_UNI_GRAM , 1)
        else:
            uni_gram_score = self.n_gram.score(preprocessed_tweet , self.ds.POS_UNI_GRAM_ITER , self.ds.NEG_UNI_GRAM_ITER ,
                                               self.ds.NEU_UNI_GRAM_ITER , 1)
            post_uni_gram_score = self.n_gram.score(pos_tag_tweet , self.ds.POS_POST_UNI_GRAM_ITER ,
                                                    self.ds.NEG_POST_UNI_GRAM_ITER ,
                                                    self.ds.NEU_POST_UNI_GRAM_ITER , 1)
        vector.append(self.mb.emoticon_score(tweet))
        vector.append(self.mb.unicode_emoticon_score(tweet))
        vector.extend(self.ws.writing_style_vector(tweet))
        vector.extend(uni_gram_score)
        vector.extend(post_uni_gram_score)

        return vector

    def map_tweet_feature_values(self, tweet):
        vector = self.lexicon._all_in_lexicon_score_(tweet)
        return vector

    def load_iteration_dict(self , is_iteration):
        if len(self.ds.UNLABELED_DICT) > 0:

            temp_pos_dict = {}
            temp_neg_dict = {}
            temp_neu_dict = {}

            for key in self.ds.UNLABELED_DICT.keys():
                tweet , last_label = self.ds.UNLABELED_DICT.get(key)
                nl = self.predict_for_iteration(tweet , last_label , is_iteration)
                if nl == self.config.LABEL_POSITIVE:
                    temp_pos_dict[key] = tweet
                if nl == self.config.LABEL_NEGATIVE:
                    temp_neg_dict[key] = tweet
                if nl == self.config.LABEL_NEUTRAL:
                    temp_neu_dict[key] = tweet
                    self.ds.UNLABELED_DICT.update({key: [ tweet , nl ]})
        else:
            temp_pos_dict = {}
            temp_neg_dict = {}
            temp_neu_dict = {}

        return temp_pos_dict , temp_neg_dict , temp_neu_dict

    def initial_run(self,test_type):
        pos , neg , neu , un_label = self.load_initial_dictionaries()
        self.ds._update_initial_dict_(pos , neg , neu , un_label , False)
        self.get_vectors_and_labels()
        self.make_model_save(False,test_type)
        return

    def iteration_run(self , is_iteration,test_type):
        pos , neg , neu = self.load_iteration_dict(is_iteration)
        self.ds._update_initial_dict_(pos , neg , neu , {} , True)
        self.get_vectors_and_labels_iteration()
        self.make_model_save(True,test_type)
        return

    def get_size(self, is_iteration):
        if is_iteration:
            pos_size = len(self.ds.POS_DICT) + len(self.ds.POS_DICT_ITER)
            neg_size = len(self.ds.NEG_DICT) + len(self.ds.NEG_DICT_ITER)
            neu_size = len(self.ds.NEU_DICT) + len(self.ds.NEU_DICT_ITER)
        else:
            pos_size = len(self.ds.POS_DICT)
            neg_size = len(self.ds.NEG_DICT)
            neu_size = len(self.ds.NEU_DICT)
        return pos_size , neg_size , neu_size
    
    def get_class_weight(self, sizes):
        pos, neg, neu = sizes
        weights = dict()
        weights[self.config.LABEL_POSITIVE] = (1.0 * neu) / pos
        weights[self.config.LABEL_NEGATIVE] = (1.0 * neu) / neg
        weights[self.config.LABEL_NEUTRAL] = 1.0
        return weights

    def do_training(self):
        for test_type in self.config.TEST_TYPES:
            self.ds.CURRENT_ITERATION = 0
            time_list = [time.time()]
            self.initial_run(test_type)
            while self.ds.CURRENT_ITERATION < 11:
                if self.ds.CURRENT_ITERATION == 0:
                    is_iteration = False
                else:
                    is_iteration = True
                self.iteration_run(is_iteration , test_type)

            time_list.append(time.time())
            print self.commons.temp_difference_cal(time_list)

    def predict(self , tweet , is_iteration):
        pass

    def generate_model(self , param , is_iteration):
        pass

    def predict_for_iteration(self , tweet , last_label , is_iteration):
        pass

    def get_vectors_and_labels(self):
        pass
    
    def get_vectors_and_labels_iteration(self):
        pass

    def make_model_save(self , param , test_type):
        pass

    def map_tweet(self , line , mode , is_iteration):
        pass
