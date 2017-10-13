# Twitter Sentiment Analysis 

## Pre-requisites
You need to satisfy following pre-requisites
 1. Python 2.7
 2. Scikit-Learn 
 3. NLTK
 
## Guide to Use
 
Initially set the path to terminal to {project}/{src} which generally ../TSAwithSSL/src
then do the following command in terminal which will give a basic python tkinter based GUI
application.
   
        python Visualizer.py
        
 
 Or otherwise if you feel to do customized, create a file inside {src}/ folder
 and name it as test.py, and add the following code.
 
  
```python
from SelfTraining import SelfTraining
method_new = SelfTraining(label,unlabel,test)
# You can fill the label,unlabel and test with 
# any possible integer values greater than 100
method_new.do_training()

or

from CoTraining import CoTraining
method_new = CoTraining(label,unlabel,test)
# You can fill the label,unlabel and test with 
# any possible integer values greater than 100
method_new.do_training()
```
and run the file using command prompt using,
            
                python test.py
look into your terminal it displays the results.