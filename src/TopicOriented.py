import time

from SelfTraining import SelfTraining


class TopicOriented(SelfTraining):
    def __init__(self , label , un_label , test):
        SelfTraining.__init__(self , label , un_label , test)
        self.NO_OF_MODELS = 1
        self.final_file = '../dataset/analysed/topic_based_' + self.get_file_prefix() + str(time.time())

    def do_training(self):
        test_type = self.config.TEST_TYPE_TWITTER_2013
        self.load_training_dictionary()
        self.update_with_topics()
        self.generate_vectors_and_labels()
        self.make_model(self.config.NO_TOPIC)
        self.save_result(test_type)
        while self.ds.CURRENT_ITERATION < 1:
            self.load_iteration_dict()
            for topic in self.ds.TOPICS.keys():
                self.generate_vectors_and_labels()
                self.make_model(topic)
            self.save_result(test_type)

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
        for i in range(self.config.NO_OF_TOPICS):
            maximum , key = self.commons.find_max_value_in_dict(original_topics)
            del original_topics[key]
            final_topics.update({key: maximum})
        final_topics.update({self.config.NO_TOPIC : 0})
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
        z_0 = self.transform_tweet(tweet , 0)
        predict_proba_0 = {}
        f_p = {}
        for topic in self.ds.TOPICS.keys():
            predict_proba_0[ topic ] = self.ds._get_model_(0 , topic).predict_proba([ z_0 ]).tolist()[ 0 ]
            f_p[topic] = max(predict_proba_0[ topic ])
        f_p_max , f_p_key = self.commons.find_max_value_in_dict(f_p)
        f_p_max_label = self.commons.get_labels(f_p_max , predict_proba_0[ f_p_key ])

        return f_p_max_label
