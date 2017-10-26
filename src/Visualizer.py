import sys

from Training import *
from SystemStore import *
from PyQt4 import QtGui , QtCore


class Visualizer(QtGui.QMainWindow):
    def _get_configuration_(self):
        try:
            label = int(self.label_limit_text.text())
        except ValueError:
            label = 10
        try:
            un_label = int(self.un_label_limit_text.text())
        except ValueError:
            un_label = 10
        try:
            test = int(self.test_limit.text())
        except ValueError:
            test = 10
        try:
            iteration = int(self.no_of_iteration.text())
        except ValueError:
            iteration = 1
        training_type = str(self.training_type_selector.currentText())
        if training_type not in self.cons.TRAINING_TYPES:
            training_type = self.cons.TRAINING_TYPES[ 0 ]
            self.training_type_selector.setEditText(training_type)
        train_type = str(self.training_type_selector.currentText())
        if train_type not in self.cons.TRAIN_TYPES:
            train_type = self.cons.TRAIN_TYPES[0]
            self.training_type_selector.setEditText(train_type)
        test_type = str(self.test_type_selector.currentText())
        if test_type not in self.cons.TEST_TYPES:
            test_type = self.cons.TEST_TYPES[0]
            self.test_type_selector.setEditText(test_type)
        try:
            confidence = 0.01 * int(self.slider_confidence.value())
        except ValueError:
            confidence = 0.50
        try:
            confidence_diff = 0.01 * int(self.slider_confidence_diff.value())
        except ValueError:
            confidence_diff = 0.50

        if training_type == self.cons.TOPIC_BASED_TRAINING_TYPE:
            self.method = TopicOriented(label , un_label , test , iteration, train_type, test_type, confidence, confidence_diff)
        elif training_type == self.cons.SELF_TRAINING_TYPE:
            self.method = SelfTraining(label , un_label , test , iteration,train_type, test_type, confidence, confidence_diff)
        elif training_type == self.cons.CO_TRAINING_TYPE:
            self.method = CoTraining(label , un_label , test , iteration,train_type, test_type, confidence, confidence_diff)

    def _generate_model_(self):
        self._get_configuration_()
        self.model_generated = False
        self.method.do_training()
        self.model_generated = True
        return True

    def _predict_tweet_(self):
        tweet = self.tweet_input.text()
        if self.model_generated:
            self.prediction.setText("Prediction is " + self.method.label_to_string(
                self.method.predict(tweet)))
        else:
            self.prediction.setText("No Model Available")

    def _config_change_(self):
        confidence_val = int(self.slider_confidence.value())
        confidence_text = "Confidence is " + str((0.01 * confidence_val))
        self.slider_confidence_diff.setMaximum(confidence_val - 34)
        self.slider_confidence.setToolTip(confidence_text)
        self.confidence_val.setText(confidence_text)

    def _config_diff_change_(self):
        confidence_diff_val = int(self.slider_confidence_diff.value())
        confidence_diff_text = "Difference is " + str((0.01 * confidence_diff_val))
        self.slider_confidence_diff.setToolTip(confidence_diff_text)
        self.confidence_val_diff.setText(confidence_diff_text)

    def __init__(self):
        super(Visualizer , self).__init__()
        self.cons = Constants()
        self.model_generated = False
        self.label_tts = QtGui.QLabel(self)
        self.label_tts.setText("Select the Training")
        self.label_tts.move(60 , 20)
        self.label_tts.resize(200, 30)
        self.training_type_selector = QtGui.QComboBox(self)
        self.training_type_selector.insertItems(1,self.cons.TRAINING_TYPES)
        self.training_type_selector.move(335, 20)
        self.training_type_selector.resize(220, 30)
        self.training_type_selector.setStyleSheet("background-color: white")
        self.label_ttsa = QtGui.QLabel(self)
        self.label_ttsa.setText("Select the Train Type")
        self.label_ttsa.move(60 , 60)
        self.label_ttsa.resize(200, 30)
        self.train_type_selector = QtGui.QComboBox(self)
        self.train_type_selector.insertItems(1,self.cons.TRAIN_TYPES)
        self.train_type_selector.move(335, 60)
        self.train_type_selector.resize(220, 30)
        self.train_type_selector.setStyleSheet("background-color: white")
        self.label_tts = QtGui.QLabel(self)
        self.label_tts.setText("Select the Test Type")
        self.label_tts.move(60 , 100)
        self.label_tts.resize(200, 30)
        self.test_type_selector = QtGui.QComboBox(self)
        self.test_type_selector.insertItems(1,self.cons.TEST_TYPES)
        self.test_type_selector.move(335, 100)
        self.test_type_selector.resize(220, 30)
        self.test_type_selector.setStyleSheet("background-color: white")
        self.label_llt = QtGui.QLabel(self)
        self.label_llt.setText("Label Limit")
        self.label_llt.move(60 , 150)
        self.label_limit_text = QtGui.QLineEdit(self)
        self.label_limit_text.move(160 , 150)
        self.label_limit_text.setText(str(10000))
        self.label_limit_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_limit_text.setStyleSheet("background-color: white")
        self.label_llt2 = QtGui.QLabel(self)
        self.label_llt2.setText("Un Label Limit")
        self.label_llt2.move(335 , 150)
        self.un_label_limit_text = QtGui.QLineEdit(self)
        self.un_label_limit_text.move(460 , 150)
        self.un_label_limit_text.setText(str(20000))
        self.un_label_limit_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.un_label_limit_text.setStyleSheet("background-color: white")
        self.label_llt3 = QtGui.QLabel(self)
        self.label_llt3.setText("Test Limit")
        self.label_llt3.move(60 , 190)
        self.test_limit = QtGui.QLineEdit(self)
        self.test_limit.move(160 , 190)
        self.test_limit.setText(str(5000))
        self.test_limit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.test_limit.setStyleSheet("background-color: white")
        self.label_llt2 = QtGui.QLabel(self)
        self.label_llt2.setText("No of Iteration")
        self.label_llt2.move(335 , 190)
        self.no_of_iteration = QtGui.QLineEdit(self)
        self.no_of_iteration.move(460 , 190)
        self.no_of_iteration.setText(str(10))
        self.no_of_iteration.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.no_of_iteration.setStyleSheet("background-color: white")
        self.slider_confidence = QtGui.QSlider(QtCore.Qt.Horizontal , self)
        self.slider_confidence.setMinimum(34)
        self.slider_confidence.setMaximum(100)
        self.slider_confidence.setValue(90)
        self.slider_confidence.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.slider_confidence.setTickInterval(5)
        self.slider_confidence.valueChanged.connect(self._config_change_)
        self.slider_confidence.move(210 , 240)
        self.slider_confidence.resize(200 , 30)
        self.slider_confidence.setStyleSheet("color: #1dcaff;")
        self.confidence = QtGui.QLabel(self)
        self.confidence.setText("Set Confidence")
        self.confidence.move(60 , 240)
        self.confidence.resize(140, 30)
        self.confidence_val = QtGui.QLabel(self)
        self.confidence_val.setText("Confidence is 0.9")
        self.confidence_val.move(440 , 240)
        self.confidence_val.resize(200, 30)
        self.confidence_val.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.slider_confidence_diff = QtGui.QSlider(QtCore.Qt.Horizontal , self)
        self.slider_confidence_diff.setMinimum(0)
        self.slider_confidence_diff.setMaximum(56)
        self.slider_confidence_diff.setValue(10)
        self.slider_confidence_diff.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.slider_confidence_diff.setTickInterval(5)
        self.slider_confidence_diff.valueChanged.connect(self._config_diff_change_)
        self.slider_confidence_diff.move(210 , 290)
        self.slider_confidence_diff.resize(200 , 30)
        self.slider_confidence_diff.setStyleSheet("color: #1dcaff;")
        self.confidence_diff = QtGui.QLabel(self)
        self.confidence_diff.setText("Set Difference")
        self.confidence_diff.move(60 , 290)
        self.confidence_diff.resize(140 , 30)
        self.confidence_val_diff = QtGui.QLabel(self)
        self.confidence_val_diff.setText("Difference is 0.1")
        self.confidence_val_diff.move(440 , 290)
        self.confidence_val_diff.resize(200 , 30)
        self.confidence_val_diff.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.generate_model = QtGui.QPushButton("Generate Model",self)
        self.generate_model.move(225 , 340)
        self.generate_model.resize(180, 50)
        self.generate_model.clicked.connect(lambda : self._generate_model_())
        self.generate_model.setStyleSheet("background-color:#383a39; color: #1dcaff;")
        self.generate_model.setToolTip("Click here to generate Model")
        self.tweet_input = QtGui.QLineEdit(self)
        self.tweet_input.move(60 , 410)
        self.tweet_input.resize(500, 30)
        self.tweet_input.setText("Welcome to Twitter!!!")
        self.tweet_input.setStyleSheet("background-color: white")
        self.tweet_input.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.tweet_input.setToolTip("Enter tweet to test")
        self.predict_tweet = QtGui.QPushButton("Predict Tweet",self)
        self.predict_tweet.move(225 , 460)
        self.predict_tweet.resize(180, 50)
        self.predict_tweet.clicked.connect(lambda : self._predict_tweet_())
        self.predict_tweet.setStyleSheet("background-color: #383a39; color: #1dcaff;")
        self.predict_tweet.setToolTip("Predict the tweet")
        self.prediction = QtGui.QLabel(self)
        self.prediction.setText("No Prediction")
        self.prediction.move(60 , 510)
        self.prediction.resize(180, 30)
        self.setGeometry(375 , 100  , 620  , 540)
        self.setWindowTitle('TSAwithSSL, v0.1.2.5_8')
        self.setStyleSheet("background-color: #1dcaff; color: #383a39;")
        self.setWindowIcon(QtGui.QIcon('../resource/images/icons/ico.png'))
        self.show()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = Visualizer()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


