import csv
import warnings
import time
from Configuration import Configuration
from Commons import Commons
from DataStore import DataStore
from Features import MicroBlog , Lexicon , WritingStyle , NGram
from PreProcess import PreProcess
import numpy as np
from sklearn import preprocessing as pr , svm
warnings.filterwarnings('ignore')


class CoTraining:
    def __init__(self,label,un_label,test):
        self.LABEL_LIMIT = label
        self.UN_LABEL_LIMIT = un_label
        self.TEST_LIMIT = test
        self.POS_COUNT_LIMIT = int(self.LABEL_LIMIT * self.config.POS_RATIO)
        self.NEG_COUNT_LIMIT = int(self.LABEL_LIMIT * self.config.NEG_RATIO)
        self.NEU_COUNT_LIMIT = int(self.LABEL_LIMIT * self.config.NEU_RATIO)
        self.ds = DataStore()
        self.config = Configuration()
        self.ppros = PreProcess()
        self.commons = Commons(self.config)
        self.mb = MicroBlog()
        self.lexicon = Lexicon(self.ppros)
        self.ws = WritingStyle()
        self.ngram = NGram(self.commons , self.ppros)
        self.final_file = '../dataset/analysed/iteration_' + self.commons.get_file_prefix() + str(time.time()) 

    def load_initial_dictionaries(self):
        """
            This used to classify initial data set as positive,negative and neutral
            :return: It return the success or failure
            """
        print self.POS_COUNT_LIMIT , self.NEG_COUNT_LIMIT , self.NEU_COUNT_LIMIT
        pos_dict = {}
        neg_dict = {}
        neu_dict = {}
        un_label_dict = {}
        with open(self.config.FILE_LABELED, 'r') as main_dataset:
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

        with open(self.config.FILE_UN_LABELED, 'r') as main_dataset:
            unlabeled = csv.reader(main_dataset)
            un_label_count = 1
            for line in unlabeled:
                if un_label_count <= self.UN_LABEL_LIMIT:
                    un_label_dict.update({str(un_label_count): [ str(line[ 0 ]) , self.config.UNLABELED ]})
                    un_label_count += 1

        return pos_dict , neg_dict , neu_dict , un_label_dict

    def map_tweet(self , tweet , mode , is_iteration):
        """
        :param tweet: 
        :param mode: 
        :param is_iteration: 
        :return: 
        """
        if mode:
            return self.map_tweet_feature_values(tweet)
        if not mode:
            return self.map_tweet_n_gram_values(tweet , is_iteration)

    def map_tweet_n_gram_values(self , tweet , is_iteration):
        """
        :param tweet: 
        :param is_iteration: 
        :return: 
        """
        vector = [ ]

        preprocessed_tweet = self.ppros.pre_process_tweet(tweet)
        pos_tag_tweet = self.ppros.pos_tag_string(preprocessed_tweet)

        if not is_iteration:
            uni_gram_score = self.ngram.score(preprocessed_tweet , self.ds.POS_UNI_GRAM , self.ds.NEG_UNI_GRAM ,
                                              self.ds.NEU_UNI_GRAM , 1)
            post_uni_gram_score = self.ngram.score(pos_tag_tweet , self.ds.POS_POST_UNI_GRAM ,
                                                   self.ds.NEG_POST_UNI_GRAM ,
                                                   self.ds.NEU_POST_UNI_GRAM , 1)
        else:
            uni_gram_score = self.ngram.score(preprocessed_tweet , self.ds.POS_UNI_GRAM_CO , self.ds.NEG_UNI_GRAM_CO ,
                                              self.ds.NEU_UNI_GRAM_CO , 1)
            post_uni_gram_score = self.ngram.score(pos_tag_tweet , self.ds.POS_POST_UNI_GRAM_CO ,
                                                   self.ds.NEG_POST_UNI_GRAM_CO ,
                                                   self.ds.NEU_POST_UNI_GRAM_CO , 1)
        vector.append(self.mb.emoticon_score(tweet))
        vector.append(self.mb.unicode_emoticon_score(tweet))
        vector.extend(self.ws.writing_style_vector(tweet))
        vector.extend(uni_gram_score)
        vector.extend(post_uni_gram_score)

        return vector

    def map_tweet_feature_values(self, tweet):
        vector = self.lexicon._all_in_lexicon_score_(tweet)
        return vector

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

    def get_vectors_and_labels(self):

        pos , pos_p = self.ngram.generate_n_gram_dict(self.ds.POS_DICT , 1)
        neg , neg_p = self.ngram.generate_n_gram_dict(self.ds.NEG_DICT , 1)
        neu , neu_p = self.ngram.generate_n_gram_dict(self.ds.NEU_DICT , 1)

        self.ds._update_uni_gram_(pos , neg , neu , False , False)
        self.ds._update_uni_gram_(pos_p , neg_p , neu_p , True , False)

        pos_vec , pos_lab = self.load_matrix_sub(self.ds.POS_DICT , 1 , self.config.LABEL_POSITIVE , False)
        neg_vec , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT , 1 , self.config.LABEL_NEGATIVE , False)
        neu_vec , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 1 , self.config.LABEL_NEUTRAL , False)
        vectors = pos_vec + neg_vec + neu_vec
        labels = pos_lab + neg_lab + neu_lab
        self.ds._update_vectors_labels_(vectors , labels , 1 , False)

        pos_vec_0 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT , 0 , self.config.LABEL_POSITIVE , False)
        neg_vec_0 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT , 0 , self.config.LABEL_NEGATIVE , False)
        neu_vec_0 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 0 , self.config.LABEL_NEUTRAL , False)
        vectors_0 = pos_vec_0 + neg_vec_0 + neu_vec_0
        self.ds._update_vectors_labels_(vectors_0 , labels , 0 , False)

        return

    def get_vectors_and_labels_iteration(self):
        """
        obtain the vectors and labels for total self training and storing it at main store
        :return:
        """
        pos_t , pos_post_t = self.ngram.generate_n_gram_dict(self.ds.POS_DICT_CO , 1)
        neg_t , neg_post_t = self.ngram.generate_n_gram_dict(self.ds.NEG_DICT_CO , 1)
        neu_t , neu_post_t = self.ngram.generate_n_gram_dict(self.ds.NEU_DICT_CO , 1)
        self.ds.POS_UNI_GRAM_CO , is_success = self.commons.dict_update(self.ds.POS_UNI_GRAM , pos_t)
        self.ds.NEG_UNI_GRAM_CO , is_success = self.commons.dict_update(self.ds.NEG_UNI_GRAM , neg_t)
        self.ds.NEU_UNI_GRAM_CO , is_success = self.commons.dict_update(self.ds.NEU_UNI_GRAM , neu_t)
        self.ds.POS_POST_UNI_GRAM_CO , is_success = self.commons.dict_update(self.ds.POS_POST_UNI_GRAM , pos_post_t)
        self.ds.NEG_POST_UNI_GRAM_CO , is_success = self.commons.dict_update(self.ds.NEG_POST_UNI_GRAM , neg_post_t)
        self.ds.NEU_POST_UNI_GRAM_CO , is_success = self.commons.dict_update(self.ds.NEU_POST_UNI_GRAM , neu_post_t)

        pos_vec , pos_lab = self.load_matrix_sub(self.ds.POS_DICT, 1 , self.config.LABEL_POSITIVE , True)
        neg_vec , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT, 1 , self.config.LABEL_NEGATIVE , True)
        neu_vec , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 1 , self.config.LABEL_NEUTRAL , True)
        pos_vec , pos_lab = self.load_matrix_sub(self.ds.POS_DICT_CO , 1 , self.config.LABEL_POSITIVE , True)
        neg_vec , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT_CO , 1 , self.config.LABEL_NEGATIVE , True)
        neu_vec , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT_CO , 1 , self.config.LABEL_NEUTRAL , True)
        vectors = pos_vec + neg_vec + neu_vec
        labels = pos_lab + neg_lab + neu_lab
        self.ds._update_vectors_labels_(vectors , labels , 1 , True)
        pos_vec_1 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT, 0 , self.config.LABEL_POSITIVE , True)
        neg_vec_1 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT , 0 , self.config.LABEL_NEGATIVE , True)
        neu_vec_1 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 0 , self.config.LABEL_NEUTRAL , True)
        pos_vec_1 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT_CO , 0 , self.config.LABEL_POSITIVE , True)
        neg_vec_1 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT_CO , 0 , self.config.LABEL_NEGATIVE , True)
        neu_vec_1 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT_CO , 0 , self.config.LABEL_NEUTRAL , True)
        vectors_1 = pos_vec_1 + neg_vec_1 + neu_vec_1
        self.ds._update_vectors_labels_(vectors_1 , labels , 0 , True)

        return is_success

    def get_class_weight(self, sizes):
        pos, neg, neu = sizes
        weights = dict()
        weights[self.config.LABEL_POSITIVE] = (1.0 * neu) / pos
        weights[self.config.LABEL_NEGATIVE] = (1.0 * neu) / neg
        weights[self.config.LABEL_NEUTRAL] = 1.0
        return weights

    def generate_model(self, mode, is_iteration):
        """
        :param mode: 
        :param is_iteration: 
        :return: 
        """
        vectors = [ ]
        labels = [ ]
        class_weights = self.get_class_weight(self.get_size(is_iteration))
        if is_iteration:
            if mode:
                vectors = self.ds.VECTORS_CO
            if not mode:
                vectors = self.ds.VECTORS_CO_0
            labels = self.ds.LABELS_CO
        if not is_iteration:
            if mode:
                vectors = self.ds.VECTORS
            if not mode:
                vectors = self.ds.VECTORS_0
            labels = self.ds.LABELS

        classifier_type = self.config.DEFAULT_CLASSIFIER
        vectors_scaled = pr.scale(np.array(vectors))
        scaler = pr.StandardScaler().fit(vectors)
        vectors_normalized = pr.normalize(vectors_scaled , norm='l2')
        normalizer = pr.Normalizer().fit(vectors_scaled)
        vectors = vectors_normalized
        vectors = vectors.tolist()

        if classifier_type == self.config.CLASSIFIER_SVM:
            kernel_function = self.config.DEFAULT_KERNEL
            c_parameter = self.config.DEFAULT_C_PARAMETER
            gamma = self.config.DEFAULT_GAMMA_SVM
            model = svm.SVC(kernel=kernel_function , C=c_parameter ,
                            class_weight=class_weights , gamma=gamma , probability=True)
            model.fit(vectors, labels)
        else:
            model = None
        self.ds._update_model_scaler_normalizer_(model , scaler , normalizer , mode)
        return

    def transform_tweet(self , tweet , mode , is_iteration):
        z = self.map_tweet(tweet , mode , is_iteration)
        if mode:
            z_scaled = self.ds.SCALAR.transform(z)
            z = self.ds.NORMALIZER.transform([ z_scaled ])
        if not mode:
            z_scaled = self.ds.SCALAR_0.transform(z)
            z = self.ds.NORMALIZER_0.transform([ z_scaled ])
        z = z[ 0 ].tolist()

        return z

    def predict(self , tweet , is_iteration):

        z = self.transform_tweet(tweet , 1 , is_iteration)
        z_0 = self.transform_tweet(tweet , 0 , is_iteration)

        predict_proba = self.ds.MODEL.predict_proba([ z ]).tolist()[ 0 ]
        predict_proba_0 = self.ds.MODEL_0.predict_proba([ z_0 ]).tolist()[ 0 ]
        sum_predict_proba = self.commons.get_sum_proba(predict_proba , predict_proba_0)
        f_p , s_p = self.commons.first_next_max(sum_predict_proba)
        f_p_l = self.commons.get_labels(f_p , sum_predict_proba)

        predict = self.ds.MODEL.predict([ z ]).tolist()[ 0 ]
        predict_0 = self.ds.MODEL_0.predict([ z_0 ]).tolist()[ 0 ]

        if predict == predict_0:
            return predict
        else:
            if f_p - s_p < self.config.PERCENTAGE_MINIMUM_DIFF or f_p < self.config.PERCENTAGE_MINIMUM_CONF:
                maxi = max(predict , predict_0)
                mini = min(predict , predict_0)

                if maxi > 0 and mini < 0:
                    return self.config.LABEL_NEUTRAL

                if maxi > 0 and mini == 0:
                    return self.config.LABEL_POSITIVE

                if maxi == 0 and mini < 0:
                    return self.config.LABEL_NEGATIVE
            else:
                return f_p_l

    def predict_for_iteration(self, tweet , last_label , is_iteration):

        z = self.transform_tweet(tweet , 1 , is_iteration)
        z_0 = self.transform_tweet(tweet , 0 , is_iteration)

        predict_proba = self.ds.MODEL.predict_proba([z]).tolist()[ 0 ]
        predict_proba_0 = self.ds.MODEL_0.predict_proba([z_0]).tolist()[ 0 ]
        sum_predict_proba = self.commons.get_sum_proba(predict_proba , predict_proba_0)
        f_p , s_p = self.commons.first_next_max(sum_predict_proba)
        f_p_l = self.commons.get_labels(f_p , sum_predict_proba)

        predict = self.ds.MODEL.predict([z]).tolist()[0]
        predict_0 = self.ds.MODEL_0.predict([z_0]).tolist()[0]

        if predict == predict_0:
            return predict
        else:
            if f_p - s_p < self.config.PERCENTAGE_MINIMUM_DIFF or f_p < self.config.PERCENTAGE_MINIMUM_CONF:
                return last_label
            else:
                return f_p_l

    def save_result(self, is_iteration, test_type):
        actual = []
        predicted = []
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
        combined = sizes + (current_iteration,) + result
        print combined
        if is_iteration:
            self.ds.CURRENT_ITERATION += 1
            temp_file = open(self.final_file + test_type +  'result.csv',"a+")
        if not is_iteration:
            temp_file = open(self.final_file + test_type +  'result.csv',"w+")
        saving_file = csv.writer(temp_file)
        if not is_iteration:
            saving_file.writerow(self.config.CSV_HEADER)
        saving_file.writerow(combined)
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

    def load_iteration_dict(self , is_iteration):
        """
        divide the unlabelled data to do self training
        :param is_iteration:
        :return:
        """
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

    def make_model_save(self, is_iteration,test_type):
        self.generate_model(0, is_iteration)
        self.generate_model(1, is_iteration)
        self.save_result(is_iteration,test_type)

    def get_size(self, is_iteration):
        if is_iteration:
            pos_size = len(self.ds.POS_DICT) + len(self.ds.POS_DICT_CO)
            neg_size = len(self.ds.NEG_DICT) + len(self.ds.NEG_DICT_CO)
            neu_size = len(self.ds.NEU_DICT) + len(self.ds.NEU_DICT_CO)
        else:
            pos_size = len(self.ds.POS_DICT)
            neg_size = len(self.ds.NEG_DICT)
            neu_size = len(self.ds.NEU_DICT)
        return pos_size , neg_size , neu_size

    def do_training(self):
        for test_type in self.config.TEST_TYPES:
            time_list = [time.time()]
            self.initial_run(test_type)
            self.ds.CURRENT_ITERATION = 0
            while self.ds.CURRENT_ITERATION < 10:
                if self.ds.CURRENT_ITERATION == 0:
                    is_iteration = False
                else:
                    is_iteration = True
                self.iteration_run(is_iteration , test_type)

            time_list.append(time.time())
            print self.commons.temp_difference_cal(time_list)