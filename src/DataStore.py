from sklearn.externals import joblib
import os


class DataStore:
    ANALYSED_DIRECTORY = "../dataset/analysed/"
    TEMP_DIRECTORY = "../dataset/temp/"
    VECTOR_TEMP_STORE = TEMP_DIRECTORY + "vector/"
    LABEL_TEMP_STORE = TEMP_DIRECTORY + "label/"
    MODEL_TEMP_STORE = TEMP_DIRECTORY + "model/"
    SCALAR_TEMP_STORE = TEMP_DIRECTORY + "scalar/"
    NORMALIZER_TEMP_STORE = TEMP_DIRECTORY + "normalizer/"

    def __init__(self):
        self.TRAIN_DICT = {}

        self.POS_INITIAL = 0
        self.NEG_INITIAL = 0
        self.NEU_INITIAL = 0

        self.POS_SIZE = 0
        self.NEG_SIZE = 0
        self.NEU_SIZE = 0

        self.CURRENT_ITERATION = 0

        self.SUB_DIRECTORY = [self.VECTOR_TEMP_STORE,self.LABEL_TEMP_STORE,
                              self.MODEL_TEMP_STORE,self.SCALAR_TEMP_STORE,
                              self.NORMALIZER_TEMP_STORE]
        self._create_directories_()

    def _create_directories_(self):
        self._create_directory_(self.ANALYSED_DIRECTORY)
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

    def _dump_model_scaler_normalizer_(self , model , scalar , normalizer , mode):
        self._create_mode_iteration_directories(mode)
        joblib.dump(model , self.MODEL_TEMP_STORE + self._get_suffix_(mode))
        joblib.dump(scalar , self.SCALAR_TEMP_STORE + self._get_suffix_(mode))
        joblib.dump(normalizer , self.NORMALIZER_TEMP_STORE + self._get_suffix_(mode))

    def _get_vectors_(self, mode):
        return joblib.load(self.VECTOR_TEMP_STORE + self._get_suffix_(mode))

    def _get_labels_(self , mode):
        return joblib.load(self.LABEL_TEMP_STORE + self._get_suffix_(mode))

    def _get_scalar_(self,mode):
        return joblib.load(self.SCALAR_TEMP_STORE + self._get_suffix_(mode))

    def _get_normalizer_(self , mode):
        return joblib.load(self.NORMALIZER_TEMP_STORE + self._get_suffix_(mode))

    def _get_model_(self , mode):
        return joblib.load(self.MODEL_TEMP_STORE + self._get_suffix_(mode))

    def _get_suffix_(self,mode):
        suffix = str(mode) + "/" + str(self._get_current_iteration_()) + "/store.plk"
        return suffix

    def _create_directory_(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _increment_iteration_(self):
        self.CURRENT_ITERATION += 1

    def _get_current_iteration_(self):
        return self.CURRENT_ITERATION

    def _update_uni_gram_(self , pos , neg , neu , is_pos_tag):
            if is_pos_tag:
                self.POS_POST_UNI_GRAM = pos
                self.NEG_POST_UNI_GRAM = neg
                self.NEU_POST_UNI_GRAM = neu
            if not is_pos_tag:
                self.POS_UNI_GRAM = pos
                self.NEG_UNI_GRAM = neg
                self.NEU_UNI_GRAM = neu
