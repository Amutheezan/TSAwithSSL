from sklearn.externals import joblib


class DataStore:
    def __init__(self):
        self.TRAIN_DICT = {}

        self.POS_INITIAL = 0
        self.NEG_INITIAL = 0
        self.NEU_INITIAL = 0

        self.POS_SIZE = 0
        self.NEG_SIZE = 0
        self.NEU_SIZE = 0

        self.CURRENT_ITERATION = 0

        self.POS_DICT_ITER= {}
        self.NEG_DICT_ITER= {}
        self.NEU_DICT_ITER= {}

        self.POS_UNI_GRAM = {}
        self.NEG_UNI_GRAM = {}
        self.NEU_UNI_GRAM = {}

    def _update_uni_gram_(self , pos , neg , neu , is_pos_tag):
        if is_pos_tag:
            self.POS_POST_UNI_GRAM = pos
            self.NEG_POST_UNI_GRAM = neg
            self.NEU_POST_UNI_GRAM = neu
        if not is_pos_tag:
            self.POS_UNI_GRAM = pos
            self.NEG_UNI_GRAM = neg
            self.NEU_UNI_GRAM = neu

    def _update_vectors_labels_(self , vector , labels , mode):
            if mode:
                joblib.dump(vector, "../dataset/temp/vector")
                joblib.dump(labels, "../dataset/temp/label")
            if not mode:
                joblib.dump(vector, "../dataset/temp/vector0")
                joblib.dump(labels, "../dataset/temp/label0")

    def _get_vectors_(self, mode):
        if mode:
            return joblib.load("../dataset/temp/vector")
        if not mode:
            return joblib.load("../dataset/temp/vector0")

    def _get_labels_(self , mode):
        if mode:
            return joblib.load("../dataset/temp/label")
        if not mode:
            return joblib.load("../dataset/temp/label0")

    def _update_model_scaler_normalizer_(self , model , scaler , normalizer , mode):
        if mode:
            joblib.dump(model,"../dataset/temp/model")
            joblib.dump(scaler,"../dataset/temp/scalar")
            joblib.dump(normalizer,"../dataset/temp/normalizer")
        if not mode:
            joblib.dump(model,"../dataset/temp/model0")
            joblib.dump(scaler,"../dataset/temp/scalar0")
            joblib.dump(normalizer,"../dataset/temp/normalizer0")

    def _get_scalar_(self,mode):
        if mode:
            return joblib.load("../dataset/temp/scalar")
        if not mode:
            return joblib.load("../dataset/temp/scalar0")

    def _get_normalizer_(self , mode):
        if mode:
            return joblib.load("../dataset/temp/normalizer")
        if not mode:
            return joblib.load("../dataset/temp/normalizer0")

    def _get_model_(self , mode):
        if mode:
            return joblib.load("../dataset/temp/model")
        if not mode:
            return joblib.load("../dataset/temp/model0")

    def _increment_iteration_(self):
        self.CURRENT_ITERATION += 1

    def _get_current_iteration_(self):
        return self.CURRENT_ITERATION


