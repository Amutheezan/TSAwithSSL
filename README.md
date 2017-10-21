# Twitter Sentiment Analysis 

## Pre-requisites
You need to satisfy following pre-requisites
 1. Python 2.7
 2. Scikit-Learn 
 3. NLTK
 
## Guide
1. Clone the repo

        git clone https://github.com/TSAwithSSL/TSAwithSSL.git
    
2. Go to repository, and go to src folder and type the following 
    commands
    
        cd ..{REPOSITORY}/src
        python Visualaizer.py
  
3. Then you will get a GUI (which is courrently in development) it is default set to self-training
you can modify it to co-training as well as topic based by replace line 67 in Visualizer.py
        
        ```python
        self.method = SelfTraining(label, un_label, test, iteration)
        ```
        
as 

        ```python
        self.method = CoTraining(label , un_label , test, iteration)
        ```
for doing co-training.

and as 

        ```python
        self.method = TopicOriented(label , un_label , test, iteration)
        ```