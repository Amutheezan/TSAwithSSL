import warnings
import time
from Wrapper import Wrapper
warnings.filterwarnings('ignore')


class SelfTraining(Wrapper):
    def __init__(self, label, un_label, test):
        Wrapper.__init__(self, label, un_label,test)
        self.NO_OF_MODELS = 1
        self.final_file = '../dataset/analysed/self_training_' + self.get_file_prefix() + str(time.time())

    def map_tweet(self , tweet , mode):
        vector = []
        if not mode:
            vector.extend(self.map_tweet_feature_values(tweet))
            vector.extend(self.map_tweet_n_gram_values(tweet))
        return vector

    def predict(self , tweet):
        z_0 = self.transform_tweet(tweet , 0)
        predict_proba_0 = self.ds._get_model_(0).predict_proba([z_0]).tolist()[ 0 ]
        f_p , s_p = self.commons.first_next_max(predict_proba_0)
        f_p_l = self.commons.get_labels(f_p , predict_proba_0)
        predict_0 = self.ds._get_model_(0).predict([ z_0 ]).tolist()[ 0 ]

        if f_p - s_p < self.config.PERCENTAGE_MINIMUM_DIFF or\
                        f_p < self.config.PERCENTAGE_MINIMUM_CONF_SELF:
            return predict_0
        else:
            return f_p_l

    def predict_for_iteration(self, tweet , last_label):
        z_0 = self.transform_tweet(tweet , 0)
        predict_proba_0 = self.ds._get_model_(0, self.config.NO_TOPIC).predict_proba([z_0]).tolist()[ 0 ]
        f_p , s_p = self.commons.first_next_max(predict_proba_0)
        f_p_l = self.commons.get_labels(f_p , predict_proba_0)

        if f_p - s_p < self.config.PERCENTAGE_MINIMUM_DIFF or \
                        f_p < self.config.PERCENTAGE_MINIMUM_CONF_SELF:
            return self.config.UNLABELED,0
        else:
            return f_p_l,f_p

