class DataStore:
    def __init__(self):
        # Dictionaries of training set both labeled and unlabeled
        self.POS_DICT = {}
        self.NEG_DICT = {}
        self.NEU_DICT = {}
        self.UNLABELED_DICT = {}
        self.POS_DICT_ITER = {}
        self.NEG_DICT_ITER = {}
        self.NEU_DICT_ITER = {}

        # Model and relevant parameters such as vector,labels, scalar and normalizer
        self.VECTORS = []
        self.VECTORS_0 = []
        self.LABELS = []
        self.LABELS_0 = []
        self.CLASS_WEIGHTS = {}
        self.CLASS_WEIGHTS_0 = {}

        self.CURRENT_ITERATION = 0

        # Uni-gram dictionaries
        self.POS_UNI_GRAM = {}
        self.NEG_UNI_GRAM = {}
        self.NEU_UNI_GRAM = {}

        self.POS_POST_UNI_GRAM = {}
        self.NEG_POST_UNI_GRAM = {}
        self.NEU_POST_UNI_GRAM = {}

        self.POS_UNI_GRAM_ITER= {}
        self.NEG_UNI_GRAM_ITER= {}
        self.NEU_UNI_GRAM_ITER= {}

        self.POS_POST_UNI_GRAM_ITER = {}
        self.NEG_POST_UNI_GRAM_ITER = {}
        self.NEU_POST_UNI_GRAM_ITER = {}

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

    def _update_vectors_labels_class_weights_(self , vector , labels , class_weights , mode):
        if mode:
            self.VECTORS = vector
            self.LABELS = labels
            self.CLASS_WEIGHTS = class_weights
        if not mode:
            self.VECTORS_0 = vector
            self.LABELS_0 = labels
            self.CLASS_WEIGHTS_0 = class_weights

    def _update_uni_gram_(self , pos ,
                          neg , neu , is_pos_tag , is_iteration):
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

    def _increment_iteration_(self):
        self.CURRENT_ITERATION += 1

    def _get_current_iteration_(self):
        return self.CURRENT_ITERATION

    def _set_current_iteration_(self , iteration):
        self.CURRENT_ITERATION = iteration


