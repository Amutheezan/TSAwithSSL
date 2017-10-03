class Configuration:
    # Location of files to be loaded
    FILE_LABELED = "../dataset/semeval2014.csv"
    FILE_UN_LABELED = "../dataset/unlabeled.csv"
    FILE_TEST = "../dataset/test2014.csv"
    FILE_TUNE = "../dataset/tune2014.csv"

    # Constant relevant to Classifier [SVM]
    CLASSIFIER_SVM = "svm"
    KERNEL_LINEAR = "linear"
    KERNEL_RBF = "rbf"

    # Test type of SemEval 2013/2014
    TEST_TYPE_LIVE_JOURNEL = "LiveJournal2014"
    TEST_TYPE_SMS = "SMS2013"
    TEST_TYPE_TWITTER_2013 = "Twitter2013"
    TEST_TYPE_TWITTER_2014 = "Twitter2014"
    TEST_TYPE_TWITTER_SARCASM = "Twitter2014Sarcasm"

    # Label float values
    LABEL_POSITIVE = 2.0
    LABEL_NEGATIVE = -2.0
    LABEL_NEUTRAL = 0.0
    UNLABELED = -4.0

    # Label String Values
    NAME_POSITIVE = "positive"
    NAME_NEGATIVE = "negative"
    NAME_NEUTRAL = "neutral"

    # Training Set Ratio
    POS_RATIO = 0.3734
    NEG_RATIO = 0.1511
    NEU_RATIO = 0.4754

    # Full Data set size
    LABEL_DATA_SET_SIZE = 9684
    TEST_DATA_SET_SIZE = 8987

    # Constants relevant at predicting
    PERCENTAGE_MINIMUM_DIFF = 0.1
    PERCENTAGE_MINIMUM_CONF = 0.5

    DEFAULT_CLASSIFIER = CLASSIFIER_SVM

    DEFAULT_KERNEL = KERNEL_RBF
    DEFAULT_C_PARAMETER = 0.28
    DEFAULT_GAMMA_SVM = 1.0

    CSV_HEADER = [ "POS" , "NEG" , "NEU" , "CODE" , "ITER" , "ACCURACY" ,
                   "PRE-POS" , "PRE-NEG" , "PRE-NEU" , "RE-POS" , "RE-NEG" , "RE-NEU" ,
                   "F1-POS" , "F1-NEG" , "F1-AVG" ]

    def __init__(self):
        self.DEFAULT_CLASSIFIER = self.CLASSIFIER_SVM
        self._setup_()

    def _setup_(self):
        self.TEST_TYPES = [self.TEST_TYPE_LIVE_JOURNEL,self.TEST_TYPE_TWITTER_2013,
                           self.TEST_TYPE_TWITTER_2014,self.TEST_TYPE_TWITTER_SARCASM,
                           self.TEST_TYPE_SMS]

        if self.DEFAULT_CLASSIFIER == self.CLASSIFIER_SVM:
            self.DEFAULT_KERNEL = self.KERNEL_RBF
            self.DEFAULT_C_PARAMETER = 0.91
            self.DEFAULT_GAMMA_SVM = 0.03
