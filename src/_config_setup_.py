import _config_constants_ as cons

# This is SETUP interface thus we can adjust input parameters
# Parameters related to training model (some constants are important to
# look nice)

LABEL_RATIO = 0.5

TEST_LIMIT = cons.TEST_DATA_SIZE

# SVM based constants
DEFAULT_KERNEL = cons.KERNEL_RBF
DEFAULT_C_PARAMETER = 0.91
DEFAULT_GAMMA_SVM = 0.03
DEFAULT_CLASS_WEIGHTS = {cons.LABEL_POSITIVE: 1.47, cons.LABEL_NEUTRAL: 1, cons.LABEL_NEGATIVE: 3.125}

# XGBoost based constants
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_DEPTH = 3
DEFAULT_MIN_CHILD_WEIGHT = 6
DEFAULT_SILENT = 0
DEFAULT_OBJECTIVE = 'multi:softmax'
DEFAULT_SUB_SAMPLE = 0.7
DEFAULT_GAMMA_XBOOST = 0
DEFAULT_REGRESSION_ALPHA = 1e-05
DEFAULT_N_ESTIMATORS = 275
DEFAULT_COLSAMPLE_BYTREE = 0.6