import csv
import warnings

import numpy as np
from sklearn import preprocessing as pr , svm
from xgboost import XGBClassifier

import _config_constants_ as cons
import _config_controller_ as controller
import _feature_lexicon_score_ as lexicon_score
import _feature_micro_blog_score_ as micro_blog_score
import _feature_n_gram_ as ngram
import _feature_postag_ as postag
import _generic_commons_ as commons
import _global_data_store_ as ds
import _pre_process_ as ppros
import _writing_style_ as ws

warnings.filterwarnings('ignore')


def get_file_prefix():
    return "{0}_{1}_". \
        format(
        str(controller.LABEL_LIMIT) , str(controller.TEST_LIMIT)
    )


def load_initial_dictionaries():
    """
    This used to classify initial data set as positive,negative and neutral
    :return: It return the success or failure
    """
    print controller.POS_COUNT_LIMIT , controller.NEG_COUNT_LIMIT , controller.NEU_COUNT_LIMIT
    pos_dict = {}
    neg_dict = {}
    neu_dict = {}
    un_label_dict = {}
    with open("../dataset/semeval.csv" , 'r') as main_dataset:
        main = csv.reader(main_dataset)
        pos_count = 1
        neg_count = 1
        neu_count = 1
        un_label_count = 1
        count = 1
        for line in main:

            if count % 3 == 0:

                if line[ 1 ] == "positive" and pos_count <= controller.POS_COUNT_LIMIT:
                    pos_dict.update({str(pos_count): str(line[ 2 ])})
                    pos_count += 1

                elif line[ 1 ] == "positive" and pos_count > controller.POS_COUNT_LIMIT:
                    un_label_dict.update({str(un_label_count): [ str(line[ 2 ]) , cons.UNLABELED ]})
                    un_label_count += 1

                if line[ 1 ] == "negative" and neg_count <= controller.NEG_COUNT_LIMIT:
                    neg_dict.update({str(neg_count): str(line[ 2 ])})
                    neg_count += 1

                elif line[ 1 ] == "negative" and neg_count > controller.NEG_COUNT_LIMIT:
                    un_label_dict.update({str(un_label_count): [ str(line[ 2 ]) , cons.UNLABELED ]})
                    un_label_count += 1

                if line[ 1 ] == "neutral" and neu_count <= controller.NEU_COUNT_LIMIT:
                    neu_dict.update({str(neu_count): str(line[ 2 ])})
                    neu_count += 1

                elif line[ 1 ] == "neutral" and neu_count > controller.NEU_COUNT_LIMIT:
                    un_label_dict.update({str(un_label_count): [ str(line[ 2 ]) , cons.UNLABELED ]})
                    un_label_count += 1

            if count % 3 == 1:

                if un_label_count <= controller.UN_LABEL_LIMIT:
                    un_label_dict.update({str(un_label_count): [ str(line[ 2 ]) , cons.UNLABELED ]})
                    un_label_count += 1

                elif un_label_count > controller.UN_LABEL_LIMIT:

                    if line[ 1 ] == "positive" and pos_count <= controller.POS_COUNT_LIMIT:
                        pos_dict.update({str(pos_count): str(line[ 2 ])})
                        pos_count += 1

                    if line[ 1 ] == "negative" and neg_count <= controller.NEG_COUNT_LIMIT:
                        neg_dict.update({str(neg_count): str(line[ 2 ])})
                        neg_count += 1

                    if line[ 1 ] == "neutral" and neu_count <= controller.NEU_COUNT_LIMIT:
                        neu_dict.update({str(neu_count): str(line[ 2 ])})
                        neu_count += 1

            count += 1

        ds.POS_DICT = pos_dict
        ds.NEG_DICT = neg_dict
        ds.NEU_DICT = neu_dict
        ds.UNLABELED_DICT = un_label_dict

    return


def map_tweet(tweet , is_co_training):
    """
    This function use to map the tweet
    :param tweet:
    :param is_co_training:
    :return:
    """

    vector = [ ]

    preprocessed_tweet = ppros.pre_process_tweet(tweet)
    postag_tweet = postag.pos_tag_string(preprocessed_tweet)

    if not is_co_training:
        uni_gram_score = ngram.score(preprocessed_tweet , ds.POS_UNI_GRAM , ds.NEG_UNI_GRAM , ds.NEU_UNI_GRAM , 1)
        post_uni_gram_score = ngram.score(postag_tweet , ds.POS_POST_UNI_GRAM , ds.NEG_POST_UNI_GRAM ,
                                          ds.NEU_POST_UNI_GRAM , 1)
    else:
        uni_gram_score = ngram.score(preprocessed_tweet , ds.POS_UNI_GRAM_SELF , ds.NEG_UNI_GRAM_SELF ,
                                     ds.NEU_UNI_GRAM_SELF , 1)
        post_uni_gram_score = ngram.score(postag_tweet , ds.POS_POST_UNI_GRAM_SELF , ds.NEG_POST_UNI_GRAM_SELF ,
                                          ds.NEU_POST_UNI_GRAM_SELF , 1)

    lexicon_score_gen = lexicon_score.get_lexicon_score(preprocessed_tweet)
    afinn_score_96 = lexicon_score.get_afinn_99_score(preprocessed_tweet)
    afinn_score_111 = lexicon_score.get_afinn_111_score(preprocessed_tweet)
    senti_140_score = lexicon_score.get_senti140_score(preprocessed_tweet)
    n_r_c_score = lexicon_score.get_NRC_score(preprocessed_tweet)
    binliu_score = lexicon_score.get_senti_word_net_score(preprocessed_tweet)
    sentiword_score = lexicon_score.get_binliu_score(preprocessed_tweet)

    emoticon_score = micro_blog_score.emoticon_score(tweet)
    unicode_emoticon_score = micro_blog_score.unicode_emoticon_score(tweet)

    writing_style = ws.writing_style_vector(tweet)

    vector.append(afinn_score_96)
    vector.append(afinn_score_111)
    vector.append(lexicon_score_gen)
    vector.append(senti_140_score)
    vector.extend(n_r_c_score)
    vector.append(binliu_score)
    vector.append(sentiword_score)
    vector.extend(writing_style)
    vector.append(emoticon_score)
    vector.append(unicode_emoticon_score)
    vector.extend(post_uni_gram_score)
    vector.extend(uni_gram_score)
    return vector


def load_matrix_sub(process_dict , feature_set=0 , label=cons.LABEL_NEUTRAL , is_co_training=False):
    """
    :param process_dict:
    :param label:
    :param is_co_training:
    :return:
    """
    limit_t = controller.LABEL_LIMIT
    if limit_t != 0:
        keys = process_dict.keys()
        if len(keys) > 0:
            vectors = [ ]
            labels = [ ]
            for key in keys:
                line = process_dict.get(key)
                z = map_tweet(line , is_co_training)
                vectors.append(z)
                labels.append(float(label))
        else:
            vectors = [ ]
            labels = [ ]
    else:
        vectors = [ ]
        labels = [ ]
    return vectors , labels


def get_vectors_and_labels():
    ds.POS_UNI_GRAM , ds.POS_POST_UNI_GRAM = ngram.generate_n_gram_dict(file_dict=ds.POS_DICT , gram=1)
    ds.NEG_UNI_GRAM , ds.NEG_POST_UNI_GRAM = ngram.generate_n_gram_dict(file_dict=ds.NEG_DICT , gram=1)
    ds.NEU_UNI_GRAM , ds.NEU_POST_UNI_GRAM = ngram.generate_n_gram_dict(file_dict=ds.NEU_DICT , gram=1)
    pos_vec , pos_lab = load_matrix_sub(process_dict=ds.POS_DICT , feature_set=0 , label=cons.LABEL_POSITIVE ,
                                        is_co_training=False)
    neg_vec , neg_lab = load_matrix_sub(process_dict=ds.NEG_DICT , feature_set=0 , label=cons.LABEL_NEGATIVE ,
                                        is_co_training=False)
    neu_vec , neu_lab = load_matrix_sub(process_dict=ds.NEU_DICT , feature_set=0 , label=cons.LABEL_NEUTRAL ,
                                        is_co_training=False)
    ds.VECTORS = pos_vec + neg_vec + neu_vec
    ds.LABELS = pos_lab + neg_lab + neu_lab
    is_success = True
    return is_success


def get_vectors_and_labels_self():
    """
    obtain the vectors and labels for total self training and storing it at main store
    :return:
    """
    pos_t , pos_post_t = ngram.generate_n_gram_dict(ds.POS_DICT_SELF, 1)
    neg_t , neg_post_t = ngram.generate_n_gram_dict(ds.NEG_DICT_SELF , 1)
    neu_t , neu_post_t = ngram.generate_n_gram_dict(ds.NEU_DICT_SELF , 1)
    ds.POS_UNI_GRAM_SELF , is_success = commons.dict_update(ds.POS_UNI_GRAM , pos_t)
    ds.NEG_UNI_GRAM_SELF , is_success = commons.dict_update(ds.NEG_UNI_GRAM , neg_t)
    ds.NEU_UNI_GRAM_SELF , is_success = commons.dict_update(ds.NEU_UNI_GRAM , neu_t)
    ds.POS_POST_UNI_GRAM_SELF , is_success = commons.dict_update(ds.POS_POST_UNI_GRAM , pos_post_t)
    ds.NEG_POST_UNI_GRAM_SELF , is_success = commons.dict_update(ds.NEG_POST_UNI_GRAM , neg_post_t)
    ds.NEU_POST_UNI_GRAM_SELF , is_success = commons.dict_update(ds.NEU_POST_UNI_GRAM , neu_post_t)
    temp_pos_dict = ds.POS_DICT.copy()
    temp_neg_dict = ds.NEG_DICT.copy()
    temp_neu_dict = ds.NEU_DICT.copy()
    temp_pos_dict_self = ds.POS_DICT_SELF.copy()
    temp_neg_dict_self = ds.NEG_DICT_SELF.copy()
    temp_neu_dict_self = ds.NEU_DICT_SELF.copy()
    temp_pos_dict_final = {}
    temp_neg_dict_final = {}
    temp_neu_dict_final = {}
    temp_pos_dict_final.update(temp_pos_dict)
    temp_neg_dict_final.update(temp_neg_dict)
    temp_neu_dict_final.update(temp_neu_dict)
    temp_pos_dict_final.update(temp_pos_dict_self)
    temp_neg_dict_final.update(temp_neg_dict_self)
    temp_neu_dict_final.update(temp_neu_dict_self)
    pos_vec , pos_lab = load_matrix_sub(temp_pos_dict_final , 0 , cons.LABEL_POSITIVE , True)
    neg_vec , neg_lab = load_matrix_sub(temp_neg_dict_final , 0 , cons.LABEL_NEGATIVE , True)
    neu_vec , neu_lab = load_matrix_sub(temp_neu_dict_final , 0 , cons.LABEL_NEUTRAL , True)
    ds.VECTORS_SELF = pos_vec + neg_vec + neu_vec
    ds.LABELS_SELF = pos_lab + neg_lab + neu_lab
    return is_success


def get_modified_class_weight(sizes):
    pos , neg , neu , test = sizes
    weights = dict()
    weights[ cons.LABEL_POSITIVE ] = (1.0 * neu) / neg
    weights[ cons.LABEL_NEGATIVE ] = (1.0 * neu) / pos
    weights[ cons.LABEL_NEUTRAL ] = 1.0
    return weights


def generate_model(classifier_type , is_co_training=False):
    """
    generating model and storing in main data store
    :param classifier_type: 
    :param is_co_training:
    :return:
    """
    if not is_co_training:
        vectors = ds.VECTORS
        labels = ds.LABELS
    else:
        vectors = ds.VECTORS_SELF
        labels = ds.LABELS_SELF
    vectors_scaled = pr.scale(np.array(vectors))
    scaler = pr.StandardScaler().fit(vectors)
    vectors_normalized = pr.normalize(vectors_scaled , norm='l2')
    normalizer = pr.Normalizer().fit(vectors_scaled)
    vectors = vectors_normalized
    vectors = vectors.tolist()
    if classifier_type == cons.CLASSIFIER_SVM:
        kernel_function = controller.DEFAULT_KERNEL
        c_parameter = controller.DEFAULT_C_PARAMETER
        gamma = controller.DEFAULT_GAMMA_SVM
        if is_co_training:
            class_weights = get_modified_class_weight(get_size(True))
        else:
            class_weights = controller.DEFAULT_CLASS_WEIGHTS
        model = svm.SVC(kernel=kernel_function , C=c_parameter ,
                        class_weight=class_weights , gamma=gamma , probability=True)
        model.fit(vectors , labels)
    elif classifier_type == cons.CLASSIFIER_XGBOOST:
        learning_rate = controller.DEFAULT_LEARNING_RATE
        max_depth = controller.DEFAULT_MAX_DEPTH
        min_child_weight = controller.DEFAULT_MIN_CHILD_WEIGHT
        silent = controller.DEFAULT_SILENT
        objective = controller.DEFAULT_OBJECTIVE
        subsample = controller.DEFAULT_SUB_SAMPLE
        gamma = controller.DEFAULT_GAMMA_XBOOST
        reg_alpha = controller.DEFAULT_REGRESSION_ALPHA
        n_estimators = controller.DEFAULT_N_ESTIMATORS
        colsample_bytree = controller.DEFAULT_COLSAMPLE_BYTREE
        model = XGBClassifier(learning_rate=learning_rate , max_depth=max_depth ,
                              min_child_weight=min_child_weight , silent=silent ,
                              objective=objective , subsample=subsample , gamma=gamma ,
                              reg_alpha=reg_alpha , n_estimators=n_estimators ,
                              colsample_bytree=colsample_bytree , )
        vectors_a = np.asarray(vectors)
        model.fit(vectors_a , labels)
    else:
        model = None
    if classifier_type == cons.CLASSIFIER_XGBOOST:
        ds.SCALAR_1 = scaler
        ds.NORMALIZER_1 = normalizer
        ds.MODEL_1 = model
    elif classifier_type == cons.CLASSIFIER_SVM:
        ds.SCALAR = scaler
        ds.NORMALIZER = normalizer
        ds.MODEL = model
    return


def transform_tweet(tweet , classifier_type , is_co_training):
    z = map_tweet(tweet , is_co_training)
    if classifier_type == cons.CLASSIFIER_XGBOOST:
        z_scaled = ds.SCALAR_1.transform(z)
        z = ds.NORMALIZER_1.transform([ z_scaled ])
    elif classifier_type == cons.CLASSIFIER_SVM:
        z_scaled = ds.SCALAR.transform(z)
        z = ds.NORMALIZER.transform([ z_scaled ])
    z = z[ 0 ].tolist()
    return z


def predict(tweet , is_co_training):
    z = transform_tweet(tweet , cons.CLASSIFIER_SVM , is_co_training)
    z_1 = transform_tweet(tweet , cons.CLASSIFIER_XGBOOST , is_co_training)
    predict_proba = ds.MODEL.predict_proba([ z ]).tolist()[ 0 ]
    predict_proba_1 = ds.MODEL_1.predict_proba([ z_1 ]).tolist()[ 0 ]
    f_p , s_p = commons.find_first_second_max(predict_proba)
    f_p_1 , s_p_1 = commons.find_first_second_max(predict_proba_1)
    max_diff = min(f_p - s_p , f_p_1 - s_p_1)
    equality_min = (f_p == f_p_1 and f_p < cons.PERCENTAGE_MINIMUM_CONF)
    if f_p != f_p_1 or max_diff < cons.PERCENTAGE_MINIMUM_DIFF or equality_min:
        predict = ds.MODEL.predict([ z ]).tolist()[ 0 ]
        predict_1 = ds.MODEL_1.predict([ z_1 ]).tolist()[ 0 ]
        if predict_1 != predict_1:
            if predict == cons.LABEL_POSITIVE and predict_1 == cons.LABEL_NEGATIVE:
                return cons.LABEL_NEUTRAL , True
            if predict == cons.LABEL_POSITIVE and predict_1 == cons.LABEL_NEUTRAL:
                return cons.LABEL_POSITIVE , True
            if predict == cons.LABEL_NEGATIVE and predict_1 == cons.LABEL_POSITIVE:
                return cons.LABEL_NEUTRAL , True
            if predict == cons.LABEL_NEGATIVE and predict_1 == cons.LABEL_NEUTRAL:
                return cons.LABEL_NEGATIVE, True
            if predict == cons.LABEL_NEUTRAL and predict_1 == cons.LABEL_POSITIVE:
                return cons.LABEL_POSITIVE , True
            if predict == cons.LABEL_NEUTRAL and predict_1 == cons.LABEL_NEGATIVE:
                return cons.LABEL_NEGATIVE , True
        else:
            return predict , True
    else:
        if predict_proba[ 0 ] == f_p:
            return cons.LABEL_NEGATIVE , True
        if predict_proba[ 1 ] == f_p:
            return cons.LABEL_NEUTRAL , True
        if predict_proba[ 2 ] == f_p:
            return cons.LABEL_POSITIVE , True


def predict_for_self_training(tweet , last_label , is_co_training):
    z = transform_tweet(tweet , cons.CLASSIFIER_SVM , is_co_training)
    z_1 = transform_tweet(tweet , cons.CLASSIFIER_XGBOOST , is_co_training)
    predict_proba = ds.MODEL.predict_proba([ z ]).tolist()[ 0 ]
    predict_proba_1 = ds.MODEL_1.predict_proba([ z_1 ]).tolist()[ 0 ]
    f_p , s_p = commons.find_first_second_max(predict_proba)
    f_p_1 , s_p_1 = commons.find_first_second_max(predict_proba_1)
    max_diff = min(f_p - s_p , f_p_1 - s_p_1)
    equality_min = (f_p == f_p_1 and f_p < cons.PERCENTAGE_MINIMUM_CONF)
    if f_p != f_p_1 or max_diff < cons.PERCENTAGE_MINIMUM_DIFF or equality_min:
        return last_label
    else:
        if predict_proba[ 0 ] == f_p:
            return cons.LABEL_NEGATIVE
        if predict_proba[ 1 ] == f_p:
            return cons.LABEL_NEUTRAL
        if predict_proba[ 2 ] == f_p:
            return cons.LABEL_POSITIVE


def store_test(is_co_training):
    test_dict = {}
    limit = controller.TEST_LIMIT
    with open('../dataset/test.csv' , "r") as testFile:
        reader = csv.reader(testFile)
        count = 0
        for line in reader:
            line = list(line)
            tweet = line[ 2 ]
            s = line[ 1 ]
            nl , is_success = predict(tweet , is_co_training)
            test_dict.update({str(count): [ s , tweet , nl ]})
            count = count + 1
            if count >= limit:
                break
    ds.TEST_DICT = test_dict
    return


def get_result(test_dict):
    """
    :param test_dict:
    :return:
    """
    TP = TN = TNeu = FP_N = FP_Neu = FN_P = FN_Neu = FNeu_P = FNeu_N = 0
    if len(test_dict) > 0:
        dic = {'positive': cons.LABEL_POSITIVE , 'negative': cons.LABEL_NEGATIVE , 'neutral': cons.LABEL_NEUTRAL}
        for key in test_dict.keys():
            line = test_dict.get(key)
            new = str(line[ 2 ])
            old = str(dic.get(line[ 0 ]))
            if old == new:
                if new == str(cons.LABEL_POSITIVE):
                    TP += 1
                elif new == str(cons.LABEL_NEGATIVE):
                    TN += 1
                elif new == str(cons.LABEL_NEUTRAL):
                    TNeu += 1
            else:
                if new == str(cons.LABEL_POSITIVE) and old == str(cons.LABEL_NEGATIVE):
                    FP_N += 1
                elif new == str(cons.LABEL_POSITIVE) and old == str(cons.LABEL_NEUTRAL):
                    FP_Neu += 1
                elif new == str(cons.LABEL_NEGATIVE) and old == str(cons.LABEL_POSITIVE):
                    FN_P += 1
                elif new == str(cons.LABEL_NEGATIVE) and old == str(cons.LABEL_NEUTRAL):
                    FN_Neu += 1
                elif new == str(cons.LABEL_NEUTRAL) and old == str(cons.LABEL_POSITIVE):
                    FNeu_P += 1
                elif new == str(cons.LABEL_NEUTRAL) and old == str(cons.LABEL_NEGATIVE):
                    FNeu_N += 1
    else:
        print "No test data"
    accuracy = commons.get_divided_value((TP + TN + TNeu) ,
                                         (TP + TN + TNeu + FP_N + FP_Neu + FN_P + FN_Neu + FNeu_P + FNeu_N))
    pre_p = commons.get_divided_value(TP , (FP_N + FP_Neu + TP))
    pre_n = commons.get_divided_value(TN , (FN_P + FN_Neu + TN))
    pre_neu = commons.get_divided_value(TNeu , (FNeu_P + FNeu_N + TNeu))
    re_p = commons.get_divided_value(TP , (FN_P + FNeu_P + TP))
    re_n = commons.get_divided_value(TN , (FP_N + FNeu_N + TN))
    re_neu = commons.get_divided_value(TNeu , (FP_Neu + FN_Neu + TNeu))
    f_score_p = 2 * commons.get_divided_value((re_p * pre_p) , (re_p + pre_p))
    f_score_n = 2 * commons.get_divided_value((re_n * pre_n) , (re_n + pre_n))
    f_score = round((f_score_p + f_score_n) / 2 , 4)

    return accuracy , pre_p , pre_n , pre_neu , re_p , re_n , re_neu , \
           f_score_p , f_score_n , f_score


def load_iteration_dict(is_co_training):
    """
    divide the unlabelled data to do self training
    :param is_co_training:
    :return:
    """
    if len(ds.UNLABELED_DICT) > 0:

        temp_pos_dict = {}
        temp_neg_dict = {}
        temp_neu_dict = {}

        for key in ds.UNLABELED_DICT.keys():
            tweet , last_label = ds.UNLABELED_DICT.get(key)
            nl = predict_for_self_training(tweet , last_label , is_co_training)
            if nl == cons.LABEL_POSITIVE:
                temp_pos_dict[ key ] = tweet
            if nl == cons.LABEL_NEGATIVE:
                temp_neg_dict[ key ] = tweet
            if nl == cons.LABEL_NEUTRAL:
                temp_neu_dict[ key ] = tweet
            ds.UNLABELED_DICT.update({key: [ tweet , nl ]})
    else:
        temp_pos_dict = {}
        temp_neg_dict = {}
        temp_neu_dict = {}

    ds.POS_DICT_SELF = temp_pos_dict
    ds.NEG_DICT_SELF = temp_neg_dict
    ds.NEU_DICT_SELF = temp_neu_dict

    return


def initial_run():
    load_initial_dictionaries()
    get_vectors_and_labels()
    generate_model(cons.CLASSIFIER_SVM , False)
    generate_model(cons.CLASSIFIER_XGBOOST , False)
    store_test(False)
    result = get_result(ds.TEST_DICT)
    ds.CURRENT_F_SCORE = result[ len(result) - 1 ]
    size = get_size(False)
    combined_result = size + (0,) + result
    return combined_result


def co_training_run(is_co_training):
    load_iteration_dict(is_co_training)
    get_vectors_and_labels_self()
    generate_model(cons.CLASSIFIER_SVM , True)
    generate_model(cons.CLASSIFIER_XGBOOST , True)
    store_test(True)
    result = get_result(ds.TEST_DICT)
    if result[ len(result) - 1 ] < ds.CURRENT_F_SCORE:
        ds.STABILITY_BREAK -= 1
        ds.CURRENT_F_SCORE = result[ len(result) - 1 ]
    else:
        ds.CURRENT_F_SCORE = result[ len(result) - 1 ]
    size = get_size(True)
    ds.CURRENT_ITERATION += 1
    current_iteration = ds.CURRENT_ITERATION
    combined_result = size + (current_iteration,) + result
    return combined_result


def get_size(is_co_training):
    if is_co_training:
        pos_size = len(ds.POS_DICT) + len(ds.POS_DICT_SELF)
        neg_size = len(ds.NEG_DICT) + len(ds.NEG_DICT_SELF)
        neu_size = len(ds.NEU_DICT) + len(ds.NEU_DICT_SELF)
        test_size = len(ds.TEST_DICT)
    else:
        pos_size = len(ds.POS_DICT)
        neg_size = len(ds.NEG_DICT)
        neu_size = len(ds.NEU_DICT)
        test_size = len(ds.TEST_DICT)
    return pos_size , neg_size , neu_size , test_size
