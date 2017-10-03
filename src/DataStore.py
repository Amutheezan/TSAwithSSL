class DataStore:
    def __init__(self):
        # Initial Dictionaries for storing classified
        # files for initial train and test data as follows
        # POS_DICT, NEG_DICT, NEU_DICT, UNLABELED_DICT, TEST_DICT
        self.POS_DICT = {}
        self.NEG_DICT = {}
        self.NEU_DICT = {}
        self.UNLABELED_DICT = {}

        # Storing the vectors and labels, and there is another vector set for
        # 2nd feature set
        self.VECTORS = [ ]
        self.LABELS = [ ]
        self.VECTORS_0 = [ ]

        # Model and relevant parameters such as scalar and normalizer
        self.MODEL = None
        self.SCALAR = None
        self.NORMALIZER = None

        # Below are duplicate of co-training...
        self.POS_DICT_ITER= {}
        self.NEG_DICT_ITER= {}
        self.NEU_DICT_ITER= {}

        self.POS_UNI_GRAM = {}
        self.NEG_UNI_GRAM = {}
        self.NEU_UNI_GRAM = {}

        self.POS_POST_UNI_GRAM = {}
        self.NEG_POST_UNI_GRAM = {}
        self.NEU_POST_UNI_GRAM = {}

        self.POS_UNI_GRAM_ITER= {}
        self.NEG_UNI_GRAM_ITER= {}
        self.NEU_UNI_GRAM_ITER= {}

        self.POS_POST_UNI_GRAM_ITER= {}
        self.NEG_POST_UNI_GRAM_ITER= {}
        self.NEU_POST_UNI_GRAM_ITER= {}

        self.VECTORS_ITER= [ ]
        self.LABELS_ITER= [ ]
        self.VECTORS_ITER_0 = [ ]

        self.MODEL_0 = None
        self.SCALAR_0 = None
        self.NORMALIZER_0 = None

        self.CURRENT_ITERATION = 0

    def _update_initial_dict_(self , pos , neg , neu , un_label , is_iteration):
        if is_iteration:
            self.POS_DICT_ITER= pos
            self.NEG_DICT_ITER= neg
            self.NEU_DICT_ITER= neu
        if not is_iteration:
            self.POS_DICT = pos
            self.NEG_DICT = neg
            self.NEU_DICT = neu
            self.UNLABELED_DICT = un_label

    def _update_vectors_labels_(self , vector , labels , mode , is_iteration):
        if is_iteration:
            if mode:
                self.VECTORS_ITER= vector
            if not mode:
                self.VECTORS_ITER_0 = vector
            self.LABELS_ITER= labels
        if not is_iteration:
            if mode:
                self.VECTORS = vector
            if not mode:
                self.VECTORS_0 = vector
            self.LABELS = labels

    def _update_uni_gram_(self , pos , neg , neu , is_pos_tag , is_iteration):
        if is_iteration:
            if is_pos_tag:
                self.POS_POST_UNI_GRAM_ITER= pos
                self.NEG_POST_UNI_GRAM_ITER= neg
                self.NEU_POST_UNI_GRAM_ITER= neu
            if not is_pos_tag:
                self.POS_UNI_GRAM_ITER= pos
                self.NEG_UNI_GRAM_ITER= neg
                self.NEU_UNI_GRAM_ITER= neu
        if not is_iteration:
            if is_pos_tag:
                self.POS_POST_UNI_GRAM = pos
                self.NEG_POST_UNI_GRAM = neg
                self.NEU_POST_UNI_GRAM = neu
            if not is_pos_tag:
                self.POS_UNI_GRAM = pos
                self.NEG_UNI_GRAM = neg
                self.NEU_UNI_GRAM = neu

    def _update_model_scaler_normalizer_(self , model , scaler , normalizer , mode):
        if mode:
            self.MODEL = model
            self.SCALAR = scaler
            self.NORMALIZER = normalizer
        if not mode:
            self.MODEL_0 = model
            self.SCALAR_0 = scaler
            self.NORMALIZER_0 = normalizer

    def _increment_iteration_(self):
        self.CURRENT_ITERATION += 1

    def _get_current_iteration_(self):
        return self.CURRENT_ITERATION


