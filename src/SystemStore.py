import operator
import os
import shutil

from sklearn.externals import joblib


# 2013

# SELF_TRAINING PARAMETERS

# {'kernel': 'rbf', 'C' : 0.28, 'gamma' :1.0} 0.65

# CO_TRAINING PARAMETERS

#  Feature Set 1
#  {'kernel': 'rbf', 'C': 0.73, 'gamma': 1.02} 0.593126981136

# Feature Set 0
# {'kernel': 'rbf', 'C': 0.1, 'gamma': 0.1} 0.0628731514651


# 2016

# SELF_TRAINING PARAMETERS

# {'kernel': 'rbf', 'C' : 0.01, 'gamma' :1.29}

# CO_TRAINING PARAMETERS

#  Feature Set 1
#  {'kernel': 'rbf', 'C': 0.65, 'gamma': 0.65}

# Feature Set 0
# {'kernel': 'rbf', 'C': 1.61, 'gamma': 0.01}


class Constants:
    # Location of files to be loaded
    FILE_LABELED_2013 = "../dataset/main/train/SemEval2013.csv"
    FILE_LABELED_2016 = "../dataset/main/train/SemEval2016.csv"
    FILE_UN_LABELED = "../dataset/main/unlabeled/Unlabeled.csv"
    FILE_TUNE_2013 = "../dataset/main/tune/Development2013.csv"
    FILE_TUNE_2016 = "../dataset/main/tune/Development2016.csv"
    FILE_TEST_2013 = "../dataset/main/test/Twitter2013.csv"
    FILE_TEST_2014 = "../dataset/main/test/Twitter2014.csv"
    FILE_TEST_2015 = "../dataset/main/test/Twitter2015.csv"
    FILE_TEST_2016 = "../dataset/main/test/Twitter2016.csv"

    TRAIN_2013 = "SemEval 2013"
    TRAIN_2016 = "SemEval 2016"
    TEST_2013 = "Twitter 2013"
    TEST_2014 = "Twitter 2014"
    TEST_2015 = "Twitter 2015"
    TEST_2016 = "Twitter 2016"
    TUNE_2013 = "Development 2013"
    TUNE_2016 = "Development 2016"

    # Constant relevant to Classifier [SVM]
    CLASSIFIER_SVM = "svm"
    KERNEL_LINEAR = "linear"
    KERNEL_RBF = "rbf"

    # Normalizer
    L1_NORMALIZER = "l1"
    L2_NORMALIZER = "l2"
    MAX_NORMALIZER = "max"

    # Label float values
    LABEL_POSITIVE = 2.0
    LABEL_NEGATIVE = -2.0
    LABEL_NEUTRAL = 0.0
    UNLABELED = -4.0

    # Label String Values
    NAME_POSITIVE = "positive"
    NAME_NEGATIVE = "negative"
    NAME_NEUTRAL = "neutral"

    # Contents related constants
    TRAIN_FILE = "train_file"
    TEST_FILE = "test_file"
    TUNE_FILE = "tune_file"
    POS_RATIO = "pos_ratio"
    NEG_RATIO = "neg_ratio"
    NEU_RATIO = "neu_ratio"
    KERNEL = "kernel"
    C_SELF = "c_self"
    GAMMA_SELF = "gamma_self"
    C_0 = "c_0"
    GAMMA_0 = "gamma_0"
    C_1 = "c_1"
    GAMMA_1 = "gamma_1"
    SIZE = "size"

    # Training Set Ratio
    POS_RATIO_2013 = 0.3734
    NEG_RATIO_2013 = 0.1511
    NEU_RATIO_2013 = 0.4754

    POS_RATIO_2016 = 0.5177
    NEG_RATIO_2016 = 0.1416
    NEU_RATIO_2016 = 0.3407

    # Full Data set size
    LABEL_DATA_SET_SIZE_2013 = 9684
    LABEL_DATA_SET_SIZE_2016 = 6000

    TEST_DATA_SET_SIZE_2013 = 1853
    TEST_DATA_SET_SIZE_2014 = 3813
    TEST_DATA_SET_SIZE_2015 = 2390
    TEST_DATA_SET_SIZE_2016 = 20633

    # training type
    CO_TRAINING_TYPE = "Co-Training"
    SELF_TRAINING_TYPE = "Self-Training"

    CSV_HEADER = ["POS" , "NEG" , "NEU" , "ITER" , "ACCURACY" ,
                   "PRE-POS" , "PRE-NEG" , "PRE-NEU" , "RE-POS" , "RE-NEG" , "RE-NEU" ,
                   "F1-POS" , "F1-NEG" , "F1-AVG" ]

    def __init__(self):
        self.DEFAULT_CLASSIFIER = self.CLASSIFIER_SVM
        self._setup_()

    def _setup_(self):
        self.LABEL_TYPES = [self.LABEL_POSITIVE, self.LABEL_NEGATIVE,self.LABEL_NEUTRAL]

        self.TRAINING_TYPES = [self.CO_TRAINING_TYPE, self.SELF_TRAINING_TYPE]

        self.TRAIN_TYPES = [self.TRAIN_2013, self.TRAIN_2016]

        self.TEST_TYPES = [self.TEST_2013, self.TEST_2014, self.TEST_2015, self.TEST_2016]

        self.TUNE_TYPES = [self.TUNE_2013, self.TUNE_2016]

        self.TRAIN_2013_CONTENTS = {
            self.TRAIN_FILE: self.FILE_LABELED_2013,
            self.POS_RATIO: self.POS_RATIO_2013,
            self.NEG_RATIO: self.NEG_RATIO_2013,
            self.NEU_RATIO: self.NEU_RATIO_2013,
            self.KERNEL: self.KERNEL_RBF,
            self.C_0: 0.1,
            self.GAMMA_0: 0.1,
            self.C_1: 0.73,
            self.GAMMA_1: 1.02,
            self.C_SELF: 0.28,
            self.GAMMA_SELF: 1.00,
            self.SIZE: self.LABEL_DATA_SET_SIZE_2013
        }

        self.TRAIN_2016_CONTENTS = {
            self.TRAIN_FILE: self.FILE_LABELED_2016,
            self.POS_RATIO: self.POS_RATIO_2016,
            self.NEG_RATIO: self.NEG_RATIO_2016,
            self.NEU_RATIO: self.NEU_RATIO_2016,
            self.KERNEL: self.KERNEL_RBF,
            self.C_0: 1.61,
            self.GAMMA_0: 0.01,
            self.C_1: 0.65,
            self.GAMMA_1: 0.65,
            self.C_SELF: 0.28,
            self.GAMMA_SELF: 1.00,
            self.SIZE: self.LABEL_DATA_SET_SIZE_2016
        }

        self.TEST_2013_CONTENTS = {
            self.TEST_FILE: self.FILE_TEST_2013,
            self.SIZE: self.TEST_DATA_SET_SIZE_2013,
        }
        self.TEST_2014_CONTENTS = {
            self.TEST_FILE: self.FILE_TEST_2014,
            self.SIZE: self.TEST_DATA_SET_SIZE_2014,
        }
        self.TEST_2015_CONTENTS = {
            self.TEST_FILE: self.FILE_TEST_2015,
            self.SIZE: self.TEST_DATA_SET_SIZE_2015,
        }
        self.TEST_2016_CONTENTS = {
            self.TEST_FILE: self.FILE_TEST_2016,
            self.SIZE: self.TEST_DATA_SET_SIZE_2016,
        }

        self.TUNE_2013_CONTENTS = {
            self.TUNE_FILE: self.FILE_TUNE_2013
        }
        self.TUNE_2016_CONTENTS = {
            self.TUNE_FILE: self.FILE_TUNE_2016
        }

        self.TRAIN_SET = {
            self.TRAIN_2013: self.TRAIN_2013_CONTENTS,
            self.TRAIN_2016: self.TRAIN_2016_CONTENTS
        }

        self.TEST_SET = {
            self.TEST_2013: self.TEST_2013_CONTENTS,
            self.TEST_2014: self.TEST_2014_CONTENTS,
            self.TEST_2015: self.TEST_2015_CONTENTS,
            self.TEST_2016: self.TEST_2016_CONTENTS,
        }

        self.TUNE_SET = {
            self.TUNE_2013: self.TUNE_2013_CONTENTS,
            self.TUNE_2016: self.TUNE_2016_CONTENTS
        }


class Commons:

    def __init__(self, cons):
        self.cons = cons

    def temp_difference_cal(self,time_list):
        if len(time_list) > 1:
            final = float(time_list[len(time_list) - 1])
            initial = float(time_list[len(time_list) - 2])
            difference = final - initial
        else:
            difference = -1.0
        return difference

    def dict_update(self,original, temp):
        result = {}
        original_temp = original.copy()
        for key in temp.keys():
            global_key_value = original_temp.get(key)
            local_key_value = temp.get(key)
            if key not in original_temp.keys():
                result.update({key: local_key_value})
            else:
                result.update({key: local_key_value + global_key_value})
                del original_temp[key]
        result.update(original_temp)
        return result

    def get_divided_value(self,numerator, denominator):
        if denominator == 0:
            return 0.0
        else:
            result = numerator / (denominator * 1.0)
            return round(result, 4)

    def first_next_max(self,input_list):
        first = 0.0
        next = 0.0
        for element in input_list:
            if element > first:
                next = first
                first = element
            elif element > next:
                next = element
        return first , next

    def find_max_value_in_dict(self , dictionary):
        maximum = max(dictionary.iteritems() , key=operator.itemgetter(1))[ 1 ]
        key = max(dictionary.iteritems() , key=operator.itemgetter(1))[ 0 ]
        return maximum , key

    def get_labels(self , p , prob):
        l = self.cons.UNLABELED
        if prob[ 0 ] == p:
            l = self.cons.LABEL_NEGATIVE
        if prob[ 1 ] == p:
            l = self.cons.LABEL_NEUTRAL
        if prob[ 2 ] == p:
            l = self.cons.LABEL_POSITIVE
        return l

    def get_sum_proba(self,first , second):
        result = [ ]
        for i in range(len(first)):
            result.append(0.5 * (first[ i ] + second[ i ]))
        return result

    def get_values(self,actual, predict):
        TP = 0
        TN = 0
        TNeu = 0
        FP_N = 0
        FP_Neu = 0
        FN_P = 0
        FN_Neu = 0
        FNeu_P = 0
        FNeu_N = 0
        for i in range(len(actual)):
            a = actual[i]
            p = predict[i]
            if a == p:
                if a == self.cons.LABEL_POSITIVE:
                    TP += 1
                if a == self.cons.LABEL_NEUTRAL:
                    TNeu += 1
                if a == self.cons.LABEL_NEGATIVE:
                    TN += 1
            if a != p:
                if a == self.cons.LABEL_POSITIVE:
                    if p == self.cons.LABEL_NEGATIVE:
                        FN_P +=1
                    if p == self.cons.LABEL_NEUTRAL:
                        FNeu_P +=1
                if a == self.cons.LABEL_NEGATIVE:
                    if p == self.cons.LABEL_POSITIVE:
                        FP_N += 1
                    if p == self.cons.LABEL_NEUTRAL:
                        FNeu_N += 1
                if a == self.cons.LABEL_NEUTRAL:
                    if p == self.cons.LABEL_POSITIVE:
                        FP_Neu += 1
                    if p == self.cons.LABEL_NEGATIVE:
                        FN_Neu += 1

        accuracy = self.get_divided_value(TP+TN+TNeu, TP+TN+TNeu+FP_N+FP_Neu+FN_P+FN_Neu+FNeu_P+FNeu_N)
        pre_pos = self.get_divided_value(TP , TP + FP_Neu + FP_N)
        pre_neg = self.get_divided_value(TN , TN + FN_Neu + FN_P)
        pre_neu = self.get_divided_value(TNeu, TNeu + FNeu_P + FNeu_N)
        re_pos = self.get_divided_value(TP, TP+FNeu_P+FN_P)
        re_neg = self.get_divided_value(TN, TN+FNeu_N+FP_N)
        re_neu = self.get_divided_value(TNeu, TNeu + FN_Neu + FP_Neu )
        f1_pos = 2 * self.get_divided_value(pre_pos * re_pos , pre_pos + re_pos)
        f1_neg = 2 * self.get_divided_value(pre_neg * re_neg , pre_neg + re_neg)
        return accuracy,pre_pos,pre_neg,pre_neu, re_pos,re_neg, re_neu,f1_pos,f1_neg, 0.5 * (f1_neg + f1_pos)


class DataStore:
    ANALYSED_DIRECTORY = "../dataset/analysed/"
    TEMP_DIRECTORY = "../dataset/temp/"
    VECTOR_TEMP_STORE = TEMP_DIRECTORY + "vector/"
    LABEL_TEMP_STORE = TEMP_DIRECTORY + "label/"
    MODEL_TEMP_STORE = TEMP_DIRECTORY + "model/"
    SCALAR_TEMP_STORE = TEMP_DIRECTORY + "scalar/"
    NORMALIZER_TEMP_STORE = TEMP_DIRECTORY + "normalizer/"

    def __init__(self, cons):
        self.cons = cons
        self.TRAIN_DICT = []
        self.TUNE_DICT = {}

        self.POS_INITIAL = []
        self.NEG_INITIAL = []
        self.NEU_INITIAL = []

        self.POS_SIZE = []
        self.NEG_SIZE = []
        self.NEU_SIZE = []

        self.POS_N_GRAM = {}
        self.NEG_N_GRAM = {}
        self.NEU_N_GRAM = {}

        self.POS_POST_N_GRAM = {}
        self.NEG_POST_N_GRAM = {}
        self.NEU_POST_N_GRAM = {}

        self.CURRENT_ITERATION = 0

        self.SUB_DIRECTORY = [self.VECTOR_TEMP_STORE, self.LABEL_TEMP_STORE, self.MODEL_TEMP_STORE,self.SCALAR_TEMP_STORE,self.NORMALIZER_TEMP_STORE]
        self._create_directories_()

    def _create_directories_(self):
        self._create_directory_(self.ANALYSED_DIRECTORY)
        self._delete_directory_(self.TEMP_DIRECTORY)
        self._create_directory_(self.TEMP_DIRECTORY)
        for directory in self.SUB_DIRECTORY:
            self._create_directory_(directory)

    def _create_mode_iteration_directories(self, mode):
        for directory in self.SUB_DIRECTORY:
            mode_directory = directory + "/" + str(mode)
            iteration_directory = mode_directory + "/" + str(self._get_current_iteration_())
            self._create_directory_(mode_directory)
            self._create_directory_(iteration_directory)

    def _dump_vectors_labels_(self , vectors , labels , mode):
        self._create_mode_iteration_directories(mode)
        joblib.dump(vectors, self.VECTOR_TEMP_STORE + self._get_suffix_(mode))
        joblib.dump(labels, self.LABEL_TEMP_STORE + self._get_suffix_(mode))

    def _dump_model_scaler_normalizer_(self, model, scalar, normalizer, mode):
        self._create_mode_iteration_directories(mode)
        joblib.dump(model, self.MODEL_TEMP_STORE + self._get_suffix_(mode))
        joblib.dump(scalar, self.SCALAR_TEMP_STORE + self._get_suffix_(mode))
        joblib.dump(normalizer, self.NORMALIZER_TEMP_STORE + self._get_suffix_(mode))

    def _get_vectors_(self, mode):
        return joblib.load(self.VECTOR_TEMP_STORE + self._get_suffix_(mode))

    def _get_labels_(self, mode):
        return joblib.load(self.LABEL_TEMP_STORE + self._get_suffix_(mode))

    def _get_scalar_(self, mode):
        return joblib.load(self.SCALAR_TEMP_STORE + self._get_suffix_(mode))

    def _get_normalizer_(self, mode):
        return joblib.load(self.NORMALIZER_TEMP_STORE + self._get_suffix_(mode))

    def _get_model_(self, mode):
        return joblib.load(self.MODEL_TEMP_STORE + self._get_suffix_(mode))

    def _get_suffix_(self, mode):
        suffix = str(mode) + "/" + str(self._get_current_iteration_()) + "/store.plk"
        return suffix

    def _create_directory_(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _delete_directory_(self,directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    def _increment_iteration_(self):
        self.CURRENT_ITERATION += 1

    def _get_current_iteration_(self):
        return self.CURRENT_ITERATION

    def _update_n_gram_(self, pos, neg, neu, n_gram, mode, is_pos_tag):
            if is_pos_tag:
                self.POS_POST_N_GRAM[str(n_gram) + "___" + str(mode)] = pos
                self.NEG_POST_N_GRAM[str(n_gram) + "___" + str(mode)] = neg
                self.NEU_POST_N_GRAM[str(n_gram) + "___" + str(mode)] = neu
            if not is_pos_tag:
                self.POS_N_GRAM[str(n_gram) + "___" + str(mode)] = pos
                self.NEG_N_GRAM[str(n_gram) + "___" + str(mode)] = neg
                self.NEU_N_GRAM[str(n_gram) + "___" + str(mode)] = neu

    def _get_n_gram_(self, n_gram, mode, is_pos_tag):
        pos = {}
        neg = {}
        neu = {}
        if is_pos_tag:
            pos = self.POS_POST_N_GRAM.get(str(n_gram) + "___" + str(mode))
            neg = self.NEG_POST_N_GRAM.get(str(n_gram) + "___" + str(mode))
            neu = self.NEU_POST_N_GRAM.get(str(n_gram) + "___" + str(mode))
        if not is_pos_tag:
            pos = self.POS_N_GRAM.get(str(n_gram) + "___" + str(mode))
            neg = self.NEG_N_GRAM.get(str(n_gram) + "___" + str(mode))
            neu = self.NEU_N_GRAM.get(str(n_gram) + "___" + str(mode))
        return  pos,neg, neu
