import sys
from PyQt4 import QtGui, QtCore
from Tuning import *


class Tuner(QtGui.QMainWindow):
    def _get_configuration_(self):
        try:
            label = int(self.label_limit_text.text())
        except ValueError:
            label = 10

        train_type = str(self.train_type_selector.currentText())
        if train_type not in self.cons.TRAIN_TYPES:
            train_type = self.cons.TRAIN_TYPES[0]
            self.train_type_selector.setEditText(train_type)
        self.method = Tuning(label, train_type)

    def _tune_model_(self):
        self._get_configuration_()
        self.model_generated = False
        self.method.do_tuning()
        self.model_generated = True
        return True

    def __init__(self):
        super(Tuner, self).__init__()
        self.cons = Constants()
        self.model_generated = False
        self.label_ttsa = QtGui.QLabel(self)
        self.label_ttsa.setText("Select the Train Data Set")
        self.label_ttsa.move(60, 20)
        self.label_ttsa.resize(240, 30)
        self.train_type_selector = QtGui.QComboBox(self)
        self.train_type_selector.insertItems(1, self.cons.TRAIN_TYPES)
        self.train_type_selector.move(60, 50)
        self.train_type_selector.resize(240, 30)
        self.train_type_selector.setStyleSheet("background-color: white")
        self.label_llt = QtGui.QLabel(self)
        self.label_llt.setText("Label Limit")
        self.label_llt.move(60, 90)
        self.label_limit_text = QtGui.QLineEdit(self)
        self.label_limit_text.move(60, 130)
        self.label_limit_text.resize(240, 30)
        self.label_limit_text.setText(str(10000))
        self.label_limit_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label_limit_text.setStyleSheet("background-color: white")
        self.tune_model = QtGui.QPushButton("Tune Parameters", self)
        self.tune_model.move(60, 170)
        self.tune_model.resize(240, 30)
        self.tune_model.clicked.connect(lambda: self._tune_model_())
        self.tune_model.setStyleSheet("background-color:#383a39; color: #1dcaff;")
        self.tune_model.setToolTip("Click here to tune parameters")
        self.setGeometry(560, 300, 360, 230)
        self.setWindowTitle('Tuner, v0.1.2.5_20171102')
        self.setStyleSheet("background-color: #1dcaff; color: #383a39;")
        self.setWindowIcon(QtGui.QIcon('../resource/images/icons/ico.png'))
        self.show()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = Tuner()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
