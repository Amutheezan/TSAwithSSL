import warnings
import time
from Wrapper import Wrapper
from SystemStore import Constants
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

    def predict(self , tweet):
        z_0 = self.transform_tweet(tweet , 0,self.cons.NO_TOPIC)
        predict_proba_0 = self.ds._get_model_(0,self.cons.NO_TOPIC).predict_proba([z_0]).tolist()[ 0 ]
        f_p , s_p = self.commons.first_next_max(predict_proba_0)
        f_p_l = self.commons.get_labels(f_p , predict_proba_0)
        predict_0 = self.ds._get_model_(0,self.cons.NO_TOPIC).predict([ z_0 ]).tolist()[ 0 ]

        if f_p - s_p < self.CONFIDENCE_DIFF or f_p < self.CONFIDENCE:
            return predict_0
        else:
            return f_p_l

    def predict_for_iteration(self, tweet , last_label):
        z_0 = self.transform_tweet(tweet , 0,self.cons.NO_TOPIC)
        predict_proba_0 = self.ds._get_model_(0,self.cons.NO_TOPIC).predict_proba([z_0]).tolist()[ 0 ]
        f_p , s_p = self.commons.first_next_max(predict_proba_0)
        f_p_l = self.commons.get_labels(f_p , predict_proba_0)

        if f_p - s_p < self.CONFIDENCE_DIFF or f_p < self.CONFIDENCE:
            return self.cons.UNLABELED,0
        else:
            return f_p_l,f_p


class CoTraining(Wrapper):
    def __init__(self, label, un_label, test, iteration, train_type, test_type, confidence, confidence_diff):
        Wrapper.__init__(self, label, un_label, test, iteration, train_type,  test_type, confidence, confidence_diff)
        self.TRAINING_TYPE = self.cons.CO_TRAINING_TYPE
        self.NO_OF_MODELS = 2
        self.final_file = '../dataset/analysed/co_training_' + self.get_file_prefix() + str(time.time())

    def map_tweet(self , tweet , mode ):
        """
        :param tweet: 
        :param mode: 
        :return: 
        """
        if mode:
            return self.map_tweet_feature_values(tweet)
        if not mode:
            return self.map_tweet_n_gram_values(tweet)

    def predict(self , tweet):
        z = self.transform_tweet(tweet , 1,self.cons.NO_TOPIC)
        z_0 = self.transform_tweet(tweet , 0,self.cons.NO_TOPIC)
        predict_proba = self.ds._get_model_(1, self.cons.NO_TOPIC).predict_proba([ z ]).tolist()[ 0 ]
        predict_proba_0 = self.ds._get_model_(0, self.cons.NO_TOPIC).predict_proba([ z_0 ]).tolist()[ 0 ]
        sum_predict_proba = self.commons.get_sum_proba(predict_proba , predict_proba_0)
        f_p , s_p = self.commons.first_next_max(sum_predict_proba)
        f_p_l = self.commons.get_labels(f_p , sum_predict_proba)
        predict = self.ds._get_model_(1,self.cons.NO_TOPIC).predict([ z ]).tolist()[ 0 ]
        predict_0 = self.ds._get_model_(0,self.cons.NO_TOPIC).predict([ z_0 ]).tolist()[ 0 ]
        if predict == predict_0:
            return predict
        else:
            if f_p - s_p < self.CONFIDENCE_DIFF or f_p < self.CONFIDENCE:
                maxi = max(predict , predict_0)
                mini = min(predict , predict_0)

                if maxi > 0 and mini < 0:
                    return self.cons.LABEL_NEUTRAL

                if maxi > 0 and mini == 0:
                    return self.cons.LABEL_POSITIVE

                if maxi == 0 and mini < 0:
                    return self.cons.LABEL_NEGATIVE
            else:
                return f_p_l

    def predict_for_iteration(self, tweet , last_label):
        z = self.transform_tweet(tweet , 1,self.cons.NO_TOPIC)
        z_0 = self.transform_tweet(tweet , 0,self.cons.NO_TOPIC)
        predict_proba = self.ds._get_model_(1,self.cons.NO_TOPIC).predict_proba([z]).tolist()[ 0 ]
        predict_proba_0 = self.ds._get_model_(0,self.cons.NO_TOPIC).predict_proba([z_0]).tolist()[ 0 ]
        sum_predict_proba = self.commons.get_sum_proba(predict_proba , predict_proba_0)
        f_p , s_p = self.commons.first_next_max(sum_predict_proba)
        f_p_l = self.commons.get_labels(f_p , sum_predict_proba)
        predict = self.ds._get_model_(1,self.cons.NO_TOPIC).predict([z]).tolist()[0]
        predict_0 = self.ds._get_model_(0,self.cons.NO_TOPIC).predict([z_0]).tolist()[0]

        if predict == predict_0:
            return predict, 0.9
        else:
            if f_p - s_p < self.CONFIDENCE_DIFF or f_p < self.CONFIDENCE:
                return self.cons.UNLABELED,0
            else:
                return f_p_l,f_p


class TopicOriented(SelfTraining):
    def __init__(self, label, un_label, test, iteration, train_type,  test_type, confidence, confidence_diff):
        SelfTraining.__init__(self, label, un_label, test, iteration, train_type, test_type, confidence, confidence_diff)
        self.TRAINING_TYPE = self.cons.TOPIC_BASED_TRAINING_TYPE
        self.NO_OF_MODELS = 1
        self.final_file = '../dataset/analysed/topic_based_' + self.get_file_prefix() + str(time.time())

    def do_training(self):
        test_type = self.cons.TEST_TYPE_TWITTER_2013
        self.load_training_dictionary()
        self.update_with_topics()
        self.generate_vectors_and_labels()
        self.make_model(self.cons.NO_TOPIC)
        self.save_result()
        while self.ds.CURRENT_ITERATION < 1:
            self.load_iteration_dict()
            for topic in self.ds.TOPICS.keys():
                self.generate_vectors_and_labels()
                self.make_model(topic)
            self.save_result()

    def update_with_topics(self):
        train = self.ds.TRAIN_DICT.copy()
        original_topics = {}
        final_topics = {}
        for key in train.keys():
            tweet , last_label , last_confidence , is_labeled , last_topic , last_value \
                = train.get(key)
            if is_labeled:
                temp_topics = self.pre_process_tweet_topic(tweet)
                original_topics, is_success = self.commons.dict_update(original_topics,temp_topics)
        for key in train.keys():
            tweet , last_label , last_confidence , is_labeled , last_topic , last_value \
                = train.get(key)
            if is_labeled:
                tweet_word = tweet.split()
                for word in tweet_word:
                        if word in original_topics.keys():
                            value = original_topics.get(word)
                            if last_value < value:
                                last_value = value
                                del train[key]
                                train.update({key: [ tweet , last_label , last_confidence ,
                                                     is_labeled , word , last_value ]})
        for i in range(self.cons.NO_OF_TOPICS):
            maximum , key = self.commons.find_max_value_in_dict(original_topics)
            del original_topics[key]
            final_topics.update({key: maximum})
        final_topics.update({self.cons.NO_TOPIC : 0})
        self.ds.TRAIN_DICT = train
        self.ds.TOPICS = final_topics
        return

    def pre_process_tweet_topic(self, tweet):
        topic_pre_process = self.pre_pros.pos_tag_string(self.pre_pros.pre_process_tweet(tweet))
        temp_topics = topic_pre_process.split()
        topics = {}
        for topic in temp_topics:
            local_value = 0
            if topic.endswith("|NOU"):
                if topic[:len(topic)-4] in topics.keys():
                    local_value = topics.get(topic[:len(topic)-4])
                topics.update({topic[:len(topic)-4]:1 + local_value})
        return topics

    def predict(self , tweet):
        predict_proba_0 = {}
        f_p = {}
        for topic in self.ds.TOPICS.keys():
            z_0 = self.transform_tweet(tweet , 0, topic)
            predict_proba_0[ topic ] = self.ds._get_model_(0 , topic).predict_proba([ z_0 ]).tolist()[ 0 ]
            f_p[topic] = max(predict_proba_0[ topic ])
        f_p_max , f_p_key = self.commons.find_max_value_in_dict(f_p)
        f_p_max_label = self.commons.get_labels(f_p_max , predict_proba_0[ f_p_key ])
        return f_p_max_label

c = Constants()
s = SelfTraining(20631, 10, 10, 1, c.TRAIN_2017, c.TEST_2017, 0.1, 0.1)
s.tune_run()
