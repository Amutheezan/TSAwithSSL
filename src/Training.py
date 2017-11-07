import time
import warnings

from Wrapper import Wrapper

warnings.filterwarnings('ignore')


class SelfTraining(Wrapper):
    def __init__(self, label, un_label, test, iteration, train_type, test_type, confidence, confidence_diff):
        Wrapper.__init__(self, label, un_label, test, iteration,train_type,  test_type, confidence, confidence_diff)
        self.TRAINING_TYPE = self.cons.SELF_TRAINING_TYPE
        self.NO_OF_MODELS = 1
        self.final_file = '../dataset/analysed/self_training_' + self.get_file_prefix() + str(time.time())

    def map_tweet(self , tweet , mode):
        vector = []
        if not mode:
            vector.extend(self.map_tweet_feature_values(tweet))
            vector.extend(self.map_tweet_n_gram_values(tweet))
        return vector

    def predict(self , tweet , mode = None):
        z_0 = self.transform_tweet(tweet, 0)
        predict_proba_0 = self.ds._get_model_(0).predict_proba([z_0]).tolist()[0]
        f_p , s_p = self.commons.first_next_max(predict_proba_0)
        f_p_l = self.commons.get_labels(f_p , predict_proba_0)
        predict_0 = self.ds._get_model_(0).predict([z_0]).tolist()[0]

        if f_p_l == self.cons.LABEL_POSITIVE and  f_p >= 0.7:
                return f_p_l
        elif f_p_l == self.cons.LABEL_NEUTRAL and  f_p >= 0.8:
                return f_p_l
        elif f_p_l == self.cons.LABEL_NEGATIVE and f_p >= 0.7:
                return f_p_l
        else:
            return predict_0

    def predict_for_iteration(self, tweet, mode = None):
        z_0 = self.transform_tweet(tweet, 0)
        predict_proba_0 = self.ds._get_model_(0).predict_proba([z_0]).tolist()[0]
        f_p , s_p = self.commons.first_next_max(predict_proba_0)
        f_p_l = self.commons.get_labels(f_p , predict_proba_0)
        if f_p_l == self.cons.LABEL_POSITIVE and f_p >= 0.7:
                return f_p_l, f_p
        elif f_p_l == self.cons.LABEL_NEUTRAL and f_p >= 0.8:
                return f_p_l, f_p
        elif f_p_l == self.cons.LABEL_NEGATIVE and f_p >= 0.7:
                return f_p_l, f_p
        else:
            return self.cons.UNLABELED , 0


class CoTraining(Wrapper):
    def __init__(self, label, un_label, test, iteration, train_type, test_type, confidence, confidence_diff):
        Wrapper.__init__(self, label, un_label, test, iteration, train_type,  test_type, confidence, confidence_diff)
        self.TRAINING_TYPE = self.cons.CO_TRAINING_TYPE
        self.NO_OF_MODELS = 2
        self.final_file = '../dataset/analysed/co_training_' + self.get_file_prefix() + str(time.time())

    def map_tweet(self , tweet , mode ):
        if mode:
            return self.map_tweet_feature_values(tweet)
        if not mode:
            return self.map_tweet_n_gram_values(tweet)

    def predict(self , tweet, mode = None):
        z = self.transform_tweet(tweet, mode)
        predict_proba = self.ds._get_model_(mode).predict_proba([z]).tolist()[0]
        f_p , s_p = self.commons.first_next_max(predict_proba)
        f_p_l = self.commons.get_labels(f_p , predict_proba)
        predict = self.ds._get_model_(mode).predict([z]).tolist()[0]

        if f_p - s_p < self.CONFIDENCE_DIFF or f_p < self.CONFIDENCE:
            return predict
        else:
            return f_p_l

    def predict_for_iteration(self, tweet, mode = None):
        if mode == 1:
            mode = 0
        if mode == 0:
            mode = 1
        z = self.transform_tweet(tweet, mode)
        predict_proba = self.ds._get_model_(mode).predict_proba([z]).tolist()[0]
        f_p , s_p = self.commons.first_next_max(predict_proba)
        f_p_l = self.commons.get_labels(f_p , predict_proba)

        if f_p - s_p < self.CONFIDENCE_DIFF or f_p < self.CONFIDENCE:
            return self.cons.UNLABELED, 0
        else:
            return f_p_l,f_p