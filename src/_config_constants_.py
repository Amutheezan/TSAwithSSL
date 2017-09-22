# CONFIGURATION CONSTANTS

CLASSIFIER_SVM = "svm"
KERNEL_LINEAR = "linear"
KERNEL_RBF = "rbf"

DATA_SET_SIZE = 20633
LABEL_POSITIVE = 2.0
LABEL_NEGATIVE = -2.0
LABEL_NEUTRAL = 0.0
UNLABELED = -4.0

POS_RATIO = 0.34213
NEG_RATIO = 0.15659
NEU_RATIO = 0.50128

# HEADER OF THE SAVING FILE

CSV_HEADER = ["pos", "neg", "neu", "test", "code", "Iteration", "Accuracy",
              "Pre - Pos", "Pre - Neg", "Pre - Neu", "Re - Pos", "Re - Neg", "Re - Neu",
              "f - Pos", "f - Neg", "f - Avg"]
