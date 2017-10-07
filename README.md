# Twitter Sentiment Analysis 

## Pre-requisites
You need to satisfy following pre-requisites
 1. Python 2.7
 2. Scikit-Learn 
 3. NLTK
 
## Guide to Use
 
Initially set the path to terminal to {project}/{src} which generally ../TSAwithSSL/src
then do one of following command in terminal.


        python Visualizer.py
        
OR go with below method
  
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