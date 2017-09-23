import _config_constants_ as cons

# This is SETUP interface thus we can adjust input parameters
# Parameters related to training model (some constants are important to
# look nice)

LABEL_RATIO = 0.5

TEST_LIMIT = cons.TEST_DATA_SIZE

DEFAULT_CLASSIFIER = cons.CLASSIFIER_SVM

if DEFAULT_CLASSIFIER == cons.CLASSIFIER_SVM:
    DEFAULT_KERNEL = cons.KERNEL_RBF
    DEFAULT_C_PARAMETER = 0.91
    DEFAULT_GAMMA_SVM = 0.03
    DEFAULT_CLASS_WEIGHTS = {cons.LABEL_POSITIVE: 1.47, cons.LABEL_NEUTRAL: 1, cons.LABEL_NEGATIVE: 3.125}
