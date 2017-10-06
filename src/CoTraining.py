import warnings
import time
import numpy as np
from sklearn import preprocessing as pr , svm
from Wrapper import Wrapper
warnings.filterwarnings('ignore')


class CoTraining(Wrapper):
    def __init__(self, label, un_label, test):
        Wrapper.__init__(self, label, un_label,test)
        self.final_file = '../dataset/analysed/co_training_' + self.get_file_prefix() + str(time.time())

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

    def get_vectors_and_labels(self):

        pos , pos_p = self.n_gram.generate_n_gram_dict(self.ds.POS_DICT , 1)
        neg , neg_p = self.n_gram.generate_n_gram_dict(self.ds.NEG_DICT , 1)
        neu , neu_p = self.n_gram.generate_n_gram_dict(self.ds.NEU_DICT , 1)

        self.ds._update_uni_gram_(pos , neg , neu , False , False)
        self.ds._update_uni_gram_(pos_p , neg_p , neu_p , True , False)

        pos_vec , pos_lab = self.load_matrix_sub(self.ds.POS_DICT , 1 , self.config.LABEL_POSITIVE , False)
        neg_vec , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT , 1 , self.config.LABEL_NEGATIVE , False)
        neu_vec , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 1 , self.config.LABEL_NEUTRAL , False)
        vectors = pos_vec + neg_vec + neu_vec
        labels = pos_lab + neg_lab + neu_lab
        vectors , scalar , normalizer = self.convert_vector(vectors)
        self.ds._update_vectors_labels_scaler_normalizer_(vectors , labels , scalar , normalizer , 1)

        pos_vec_0 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT , 0 , self.config.LABEL_POSITIVE , False)
        neg_vec_0 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT , 0 , self.config.LABEL_NEGATIVE , False)
        neu_vec_0 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 0 , self.config.LABEL_NEUTRAL , False)
        vectors_0 = pos_vec_0 + neg_vec_0 + neu_vec_0
        vectors_0 , scalar , normalizer = self.convert_vector(vectors_0)
        self.ds._update_vectors_labels_scaler_normalizer_(vectors_0 , labels , scalar , normalizer , 0)
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

        pos_vec , pos_lab = self.load_matrix_sub(self.ds.POS_DICT, 1 , self.config.LABEL_POSITIVE , True)
        neg_vec , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT, 1 , self.config.LABEL_NEGATIVE , True)
        neu_vec , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 1 , self.config.LABEL_NEUTRAL , True)
        vectors = pos_vec + neg_vec + neu_vec
        labels = pos_lab + neg_lab + neu_lab
        pos_vec , pos_lab = self.load_matrix_sub(self.ds.POS_DICT_ITER, 1 , self.config.LABEL_POSITIVE , True)
        neg_vec , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT_ITER, 1 , self.config.LABEL_NEGATIVE , True)
        neu_vec , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT_ITER, 1 , self.config.LABEL_NEUTRAL , True)
        vectors = vectors + pos_vec + neg_vec + neu_vec
        labels = labels + pos_lab + neg_lab + neu_lab
        vectors , scalar , normalizer = self.convert_vector(vectors)
        self.ds._update_vectors_labels_scaler_normalizer_(vectors , labels , scalar , normalizer , 1)

        pos_vec_0 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT, 0 , self.config.LABEL_POSITIVE , True)
        neg_vec_0 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT , 0 , self.config.LABEL_NEGATIVE , True)
        neu_vec_0 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT , 0 , self.config.LABEL_NEUTRAL , True)
        vectors_0 = pos_vec_0 + neg_vec_0 + neu_vec_0
        pos_vec_0 , pos_lab = self.load_matrix_sub(self.ds.POS_DICT_ITER, 0 , self.config.LABEL_POSITIVE , True)
        neg_vec_0 , neg_lab = self.load_matrix_sub(self.ds.NEG_DICT_ITER, 0 , self.config.LABEL_NEGATIVE , True)
        neu_vec_0 , neu_lab = self.load_matrix_sub(self.ds.NEU_DICT_ITER, 0 , self.config.LABEL_NEUTRAL , True)
        vectors_0 = vectors_0 + pos_vec_0+ neg_vec_0 + neu_vec_0
        vectors_0 , scalar , normalizer = self.convert_vector(vectors_0)
        self.ds._update_vectors_labels_scaler_normalizer_(vectors_0 , labels , scalar , normalizer , 0)

        return is_success

    def make_model(self, is_iteration):
        self.generate_model(0, is_iteration)
        self.generate_model(1, is_iteration)
        if is_iteration:
            self.ds._increment_iteration_()

    def generate_model(self, mode, is_iteration):
        """
        :param mode: 
        :param is_iteration: 
        :return: 
        """
        c_parameter = 0.0
        gamma = 0.0
        class_weights = self.get_class_weight(self.get_size(is_iteration))
        if mode:
            c_parameter = self.config.DEFAULT_C_PARAMETER
            gamma = self.config.DEFAULT_GAMMA_SVM
        if not mode:
            c_parameter = self.config.DEFAULT_C_PARAMETER_0
            gamma = self.config.DEFAULT_GAMMA_SVM_0
        classifier_type = self.config.DEFAULT_CLASSIFIER
        if classifier_type == self.config.CLASSIFIER_SVM:
            kernel_function = self.config.DEFAULT_KERNEL
            model = svm.SVC(kernel=kernel_function , C=c_parameter ,
                            class_weight=class_weights , gamma=gamma , probability=True)
        else:
            model = None
        self.ds._update_model_(model , mode)
        return

    def transform_tweet(self , tweet , mode , is_iteration):
        z = self.map_tweet(tweet , mode , is_iteration)
        if mode:
            z_scaled = self.ds.SCALAR[ self.ds.CURRENT_ITERATION ].transform(z)
            z = self.ds.NORMALIZER[ self.ds.CURRENT_ITERATION ].transform([ z_scaled ])
        if not mode:
            z_scaled = self.ds.SCALAR_0[ self.ds.CURRENT_ITERATION ].transform(z)
            z = self.ds.NORMALIZER_0[ self.ds.CURRENT_ITERATION ].transform([ z_scaled ])
        z = z[ 0 ].tolist()

        return z

    def predict(self , tweet , is_iteration):
        z = self.transform_tweet(tweet , 1 , is_iteration)
        z_0 = self.transform_tweet(tweet , 0 , is_iteration)
        temp_model_0 = self.ds.MODEL_0[ self.ds.CURRENT_ITERATION ]
        temp_model_0 = temp_model_0.fit(self.ds.VECTORS_0[ self.ds.CURRENT_ITERATION ] ,
                                        self.ds.LABELS_0[ self.ds.CURRENT_ITERATION ])
        temp_model = self.ds.MODEL[ self.ds.CURRENT_ITERATION ]
        temp_model = temp_model.fit(self.ds.VECTORS[ self.ds.CURRENT_ITERATION ] ,
                                        self.ds.LABELS[ self.ds.CURRENT_ITERATION ])
        predict_proba = temp_model.predict_proba([ z ]).tolist()[ 0 ]
        predict_proba_0 = temp_model_0.predict_proba([ z_0 ]).tolist()[ 0 ]
        sum_predict_proba = self.commons.get_sum_proba(predict_proba , predict_proba_0)
        f_p , s_p = self.commons.first_next_max(sum_predict_proba)
        f_p_l = self.commons.get_labels(f_p , sum_predict_proba)
        predict = temp_model.predict([ z ]).tolist()[ 0 ]
        predict_0 = temp_model_0.predict([ z_0 ]).tolist()[ 0 ]
        del temp_model_0,temp_model
        if predict == predict_0:
            return predict
        else:
            if f_p - s_p < self.config.PERCENTAGE_MINIMUM_DIFF\
                    or f_p < self.config.PERCENTAGE_MINIMUM_CONF_CO:
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
        temp_model_0 = self.ds.MODEL_0[ self.ds.CURRENT_ITERATION ]
        temp_model_0 = temp_model_0.fit(self.ds.VECTORS_0[ self.ds.CURRENT_ITERATION ] ,
                                        self.ds.LABELS_0[ self.ds.CURRENT_ITERATION ])
        temp_model = self.ds.MODEL[ self.ds.CURRENT_ITERATION ]
        temp_model = temp_model.fit(self.ds.VECTORS[ self.ds.CURRENT_ITERATION ] ,
                                        self.ds.LABELS[ self.ds.CURRENT_ITERATION ])
        predict_proba = temp_model.predict_proba([ z ]).tolist()[ 0 ]
        predict_proba_0 = temp_model_0.predict_proba([ z_0 ]).tolist()[ 0 ]
        sum_predict_proba = self.commons.get_sum_proba(predict_proba , predict_proba_0)
        f_p , s_p = self.commons.first_next_max(sum_predict_proba)
        f_p_l = self.commons.get_labels(f_p , sum_predict_proba)
        predict = temp_model.predict([ z ]).tolist()[ 0 ]
        predict_0 = temp_model_0.predict([ z_0 ]).tolist()[ 0 ]
        del temp_model_0,temp_model
        if predict == predict_0:
            return predict
        else:
            if f_p - s_p < self.config.PERCENTAGE_MINIMUM_DIFF \
                    or f_p < self.config.PERCENTAGE_MINIMUM_CONF_CO:
                return last_label
            else:
                return f_p_l

