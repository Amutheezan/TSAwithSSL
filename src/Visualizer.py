from tkinter import *
import tkinter.messagebox as tm
from CoTraining import CoTraining
from SelfTraining import SelfTraining
from TopicOriented import TopicOriented


class MainFrame(Frame):
    def __init__(self,master):
        Frame.__init__(self)
        self.label_head = Label(self , text="Twitter Sentiment Analysis", fg="#00008b",)
        self.label_head.config(font=("Times New Roman" , 20))
        self.label_head.grid(columnspan=2)

        self.label_1 = Label(self , text="Enter the Test Tweet")
        self.entry_1 = Entry(self,justify=CENTER)
        self.label_1.grid(row=5 , sticky=E)
        self.entry_1.grid(row=6 , columnspan=8,rowspan=2)

        self.label_label = Label(self , text="Label Limit")
        self.entry_label = Entry(self,justify=RIGHT)
        self.label_label.grid(row=1 , sticky=E)
        self.entry_label.grid(row=1 , column=1)
        self.entry_label.insert(END,'10000')

        self.label_un_label = Label(self , text="Un Label Limit")
        self.entry_un_label = Entry(self,justify=RIGHT)
        self.label_un_label.grid(row=2 , sticky=E)
        self.entry_un_label.grid(row=2 , column=1)
        self.entry_un_label.insert(END,'10000')

        self.label_test = Label(self , text="Test Limit")
        self.entry_test = Entry(self,justify=RIGHT)
        self.label_test.grid(row=3 , sticky=E)
        self.entry_test.grid(row=3 , column=1)
        self.entry_test.insert(END,'10000')

        self.train_btn = Button(self , text="Train Model" ,  fg="#a1dbcd", bg="#383a39", command=self._generate_model_)
        self.train_btn.grid(row=4, column=1)
        self.predict_btn = Button(self , text="Predict Tweet" ,  fg="#a1dbcd", bg="#383a39", command=self._predict_model_)
        self.predict_btn.grid(row=8, columnspan=2)

        self.model_generated = False
        self.pack()

    def _get_configuration_(self):
        try:
            label = int(self.entry_label.get())
        except ValueError:
            label = 10
        try:
            un_label = int(self.entry_un_label.get())
        except ValueError:
            un_label = 10
        try:
            test = int(self.entry_test.get())
        except ValueError:
            test = 10
        self.method = CoTraining(label, un_label, test)

    def _generate_model_(self):
        self._get_configuration_()
        self.model_generated = False
        self.method.do_training()
        self.model_generated = True

    def _predict_model_(self):
        tweet = self.entry_1.get()
        if self.model_generated:
            tm.showinfo("Prediction","tweet is " + str(self.method.label_to_string(
                self.method.predict(tweet))))
        else:
            tm.showerror("Predict Error","No Models Available")
            self._generate_model_()
            self._predict_model_()

root = Tk()
root.title("TSAwithSSL, v0.1.1.0")
lf = MainFrame(root)
root.mainloop()
