import warnings
import time
import numpy as np
from sklearn import preprocessing as pr , svm
from Wrapper import Wrapper
warnings.filterwarnings('ignore')


class SelfTraining(Wrapper):
    def __init__(self, label, un_label, test):
        Wrapper.__init__(self, label, un_label,test)
        self.final_file = '../dataset/analysed/self_training_' + self.get_file_prefix() + str(time.time())

    def map_tweet(self , tweet , mode , is_iteration):
        vector = []
        if not mode:
            vector.extend(self.map_tweet_feature_values(tweet))
            vector.extend(self.map_tweet_n_gram_values(tweet , is_iteration))
        return vector

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
                    tweet = process_dict.get(key)
                    z = self.map_tweet(tweet , mode , is_iteration)
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

        pos , pos_p = self.n_gram.generate_n_gram_dict(self.ds.POS_DICT , 1)
        neg , neg_p = self.n_gram.generate_n_gram_dict(self.ds.NEG_DICT , 1)
        neu , neu_p = self.n_gram.generate_n_gram_dict(self.ds.NEU_DICT , 1)

        self.ds._update_uni_gram_(pos , neg , neu , False , False)
        self.ds._update_uni_gram_(pos_p , neg_p , neu_p , True , False)

        pos_vec_0 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT , 0 , self.config.LABEL_POSITIVE , False)
        neg_vec_0 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT , 0 , self.config.LABEL_NEGATIVE , False)
        neu_vec_0 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 0 , self.config.LABEL_NEUTRAL , False)
        vectors_0 = pos_vec_0 + neg_vec_0 + neu_vec_0
        labels = pos_lab + neg_lab + neu_lab
        self.ds._update_vectors_labels_(vectors_0 , labels , 0 , False)

        return

    def get_vectors_and_labels_iteration(self):
        """
        obtain the vectors and labels for total self training and storing it at main store
        :return:
        """
        pos_t , pos_post_t = self.n_gram.generate_n_gram_dict(self.ds.POS_DICT_ITER , 1)
        neg_t , neg_post_t = self.n_gram.generate_n_gram_dict(self.ds.NEG_DICT_ITER , 1)
        neu_t , neu_post_t = self.n_gram.generate_n_gram_dict(self.ds.NEU_DICT_ITER , 1)
        self.ds.POS_UNI_GRAM_ITER, is_success = self.commons.dict_update(self.ds.POS_UNI_GRAM , pos_t)
        self.ds.NEG_UNI_GRAM_ITER, is_success = self.commons.dict_update(self.ds.NEG_UNI_GRAM , neg_t)
        self.ds.NEU_UNI_GRAM_ITER, is_success = self.commons.dict_update(self.ds.NEU_UNI_GRAM , neu_t)
        self.ds.POS_POST_UNI_GRAM_ITER, is_success = self.commons.dict_update(self.ds.POS_POST_UNI_GRAM , pos_post_t)
        self.ds.NEG_POST_UNI_GRAM_ITER, is_success = self.commons.dict_update(self.ds.NEG_POST_UNI_GRAM , neg_post_t)
        self.ds.NEU_POST_UNI_GRAM_ITER, is_success = self.commons.dict_update(self.ds.NEU_POST_UNI_GRAM , neu_post_t)

        pos_vec_1 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT, 0 , self.config.LABEL_POSITIVE , True)
        neg_vec_1 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT , 0 , self.config.LABEL_NEGATIVE , True)
        neu_vec_1 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 0 , self.config.LABEL_NEUTRAL , True)
        labels = pos_lab + neg_lab + neu_lab
        vectors_1 = pos_vec_1 + neg_vec_1 + neu_vec_1
        pos_vec_1 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT_ITER, 0 , self.config.LABEL_POSITIVE , True)
        neg_vec_1 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT_ITER, 0 , self.config.LABEL_NEGATIVE , True)
        neu_vec_1 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT_ITER, 0 , self.config.LABEL_NEUTRAL , True)
        vectors_1 = vectors_1 + pos_vec_1 + neg_vec_1 + neu_vec_1
        labels = labels + pos_lab + neg_lab + neu_lab
        self.ds._update_vectors_labels_(vectors_1 , labels , 0 , True)

        return is_success

    def make_model_save(self, is_iteration,test_type):
        self.generate_model(0, is_iteration)
        self.save_result(is_iteration,test_type)

    def generate_model(self, mode, is_iteration):
        """
        :param mode: 
        :param is_iteration: 
        :return: 
        """
        vectors = []
        labels = []
        class_weights = self.get_class_weight(self.get_size(is_iteration))
        if is_iteration:
            vectors = self.ds.VECTORS_ITER_0
            labels = self.ds.LABELS_ITER
        if not is_iteration:
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
            kernel_function = self.config.DEFAULT_KERNEL_SELF
            c_parameter = self.config.DEFAULT_C_PARAMETER_SELF
            gamma = self.config.DEFAULT_GAMMA_SVM_SELF
            model = svm.SVC(kernel=kernel_function , C=c_parameter ,
                            class_weight=class_weights , gamma=gamma , probability=True)
            model.fit(vectors, labels)
        else:
            model = None
        self.ds._update_model_scaler_normalizer_(model , scaler , normalizer , mode)
        return

    def transform_tweet(self , tweet , mode , is_iteration):
        z = self.map_tweet(tweet , mode , is_iteration)
        z_scaled = self.ds.SCALAR_0.transform(z)
        z = self.ds.NORMALIZER_0.transform([ z_scaled ])
        z = z[ 0 ].tolist()
        return z

    def predict(self , tweet , is_iteration):
        z_0 = self.transform_tweet(tweet , 0 , is_iteration)
        predict_proba_0 = self.ds.MODEL_0.predict_proba([z_0]).tolist()[ 0 ]
        f_p , s_p = self.commons.first_next_max(predict_proba_0)
        f_p_l = self.commons.get_labels(f_p , predict_proba_0)
        predict_0 = self.ds.MODEL_0.predict([ z_0 ]).tolist()[ 0 ]

        if f_p - s_p < self.config.PERCENTAGE_MINIMUM_DIFF or\
                        f_p < self.config.PERCENTAGE_MINIMUM_CONF_SELF:
            return predict_0
        else:
            return f_p_l

    def predict_for_iteration(self, tweet , last_label , is_iteration):
        z_0 = self.transform_tweet(tweet , 0 , is_iteration)
        predict_proba_0 = self.ds.MODEL_0.predict_proba([z_0]).tolist()[ 0 ]
        f_p , s_p = self.commons.first_next_max(predict_proba_0)
        f_p_l = self.commons.get_labels(f_p , predict_proba_0)

        if f_p - s_p < self.config.PERCENTAGE_MINIMUM_DIFF or \
                        f_p < self.config.PERCENTAGE_MINIMUM_CONF_SELF:
            return last_label
        else:
            return f_p_l

