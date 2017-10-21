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
        train_type = str(self.training_type_selector.currentText())
        if train_type not in self.cons.TRAINING_TYPES:
            train_type = self.cons.TRAINING_TYPES[ 0 ]
            self.training_type_selector.setEditText(train_type)
        test_type = str(self.test_type_selector.currentText())
        if test_type not in self.cons.TEST_TYPES:
            test_type = self.cons.TEST_TYPES[0]
            self.test_type_selector.setEditText(test_type)
        if train_type == self.cons.TOPIC_BASED_TRAINING_TYPE:
            self.method = TopicOriented(label , un_label , test , iteration,test_type)
        elif train_type == self.cons.SELF_TRAINING_TYPE:
            self.method = SelfTraining(label , un_label , test , iteration,test_type)
        elif train_type == self.cons.CO_TRAINING_TYPE:
            self.method = CoTraining(label , un_label , test , iteration,test_type)

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
            self._generate_model_()
            self._predict_tweet_()

    def __init__(self):
        super(Visualizer , self).__init__()
        self.cons = Constants()
        self.model_generated = False
        self.label_tts = QtGui.QLabel(self)
        self.label_tts.setText("Select the Training")
        self.label_tts.move(60 , 20)
        self.label_tts.resize(200, 20)
        self.training_type_selector = QtGui.QComboBox(self)
        self.training_type_selector.insertItems(1,self.cons.TRAINING_TYPES)
        self.training_type_selector.move(60,50)
        self.training_type_selector.resize(200, 30)
        self.label_tts = QtGui.QLabel(self)
        self.label_tts.setText("Select the Test Type")
        self.label_tts.move(335 , 20)
        self.label_tts.resize(200, 20)
        self.test_type_selector = QtGui.QComboBox(self)
        self.test_type_selector.insertItems(1,self.cons.TEST_TYPES)
        self.test_type_selector.move(335, 50)
        self.test_type_selector.resize(220, 30)
        self.label_llt = QtGui.QLabel(self)
        self.label_llt.setText("Label Limit")
        self.label_llt.move(60 , 100)
        self.label_limit_text = QtGui.QLineEdit(self)
        self.label_limit_text.move(160 , 100)
        self.label_limit_text.setText(str(10000))
        self.label_limit_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_llt2 = QtGui.QLabel(self)
        self.label_llt2.setText("Un Label Limit")
        self.label_llt2.move(335 , 100)
        self.un_label_limit_text = QtGui.QLineEdit(self)
        self.un_label_limit_text.move(460 , 100)
        self.un_label_limit_text.setText(str(20000))
        self.un_label_limit_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_llt3 = QtGui.QLabel(self)
        self.label_llt3.setText("Test Limit")
        self.label_llt3.move(60 , 140)
        self.test_limit = QtGui.QLineEdit(self)
        self.test_limit.move(160 , 140)
        self.test_limit.setText(str(5000))
        self.test_limit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_llt2 = QtGui.QLabel(self)
        self.label_llt2.setText("No of Iteration")
        self.label_llt2.move(335 , 140)
        self.no_of_iteration = QtGui.QLineEdit(self)
        self.no_of_iteration.move(460 , 140)
        self.no_of_iteration.setText(str(10))
        self.no_of_iteration.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.generate_model = QtGui.QPushButton("Generate Model",self)
        self.generate_model.move(225 , 190)
        self.generate_model.resize(180, 50)
        self.generate_model.clicked.connect(lambda : self._generate_model_())
        self.generate_model.setStyleSheet("background-color:#383a39; color: #a1dbcd;")
        self.tweet_input = QtGui.QLineEdit(self)
        self.tweet_input.move(60 , 260)
        self.tweet_input.resize(500, 30)
        self.tweet_input.setText("Welcome to Twitter!!!")
        self.tweet_input.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.predict_tweet = QtGui.QPushButton("Predict Tweet",self)
        self.predict_tweet.move(225 , 310)
        self.predict_tweet.resize(180, 50)
        self.predict_tweet.clicked.connect(lambda : self._predict_tweet_())
        self.predict_tweet.setStyleSheet("background-color: #383a39; color: #a1dbcd;")
        self.prediction = QtGui.QLabel(self)
        self.prediction.setText("No Prediction")
        self.prediction.move(30 , 360)
        self.setGeometry(375 , 200  , 620  , 400)
        self.setWindowTitle('TSAwithSSL, v0.1.2.3')
        self.setStyleSheet("background-color: #a1dbcd; color: #383a39;")
        self.show()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = Visualizer()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


