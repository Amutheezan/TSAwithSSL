from flask import *
from SelfTraining import SelfTraining
from CoTraining import CoTraining
app = Flask(__name__)
is_trained = False


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/training' , methods=['POST'])
def training():
  global is_trained,method_new
  is_trained = False
  label = request.form['LabelLimit']
  un_label = request.form['UnLabelLimit']
  text = request.form['TestLimit']
  method_new = _get_configuration_(label,un_label,text)
  method_new.do_training()
  print 'Training Completed'
  is_trained = True
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  if is_trained:
    tweet = request.form[ 'tweet' ]
    result =method_new.predict(tweet,True)
    lbl_result = 'Prediction for ' + tweet + " is " + method_new.label_to_string(result)
    print lbl_result
    return render_template('index.html',lbl_result=lbl_result)
  else:
    lbl_result = 'No Model Generated Yet'
    print lbl_result
    return render_template('index.html' , lbl_result=lbl_result)

def _get_configuration_(label,un_label,test):
  try:
    label = int(label)
  except ValueError:
    label = 100
  try:
    un_label = int(un_label)
  except ValueError:
    un_label = 100
  try:
    test = int(test)
  except ValueError:
    test = 100
  return SelfTraining(label , un_label , test)

if __name__ == '__main__':
  app.run(debug=True)