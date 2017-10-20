from SelfTraining import SelfTraining
import time


class TopicOriented(SelfTraining):
    def __init__(self , label , un_label , test):
        SelfTraining.__init__(self , label , un_label , test)
        self.NO_OF_MODELS = 1
        self.final_file = '../dataset/analysed/topic_based_' + self.get_file_prefix() + str(time.time())

    def do_training(self):
        self.load_training_dictionary()
        self.update_with_topics()
        for topic in self.ds.TOPICS.keys():
            self.generate_vectors_and_labels()
            self.make_model(topic)
        print "Models making finished successfully"

    def update_with_topics(self):
        train = self.ds.TRAIN_DICT.copy()
        original_topics = {}
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
        self.ds.TRAIN_DICT = train
        self.ds.TOPICS = original_topics
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
        for topic in topics.keys():
            value = topics.get(topic)
            if value < 3:
                del topics[topic]
        return topics