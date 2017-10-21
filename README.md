# Twitter Sentiment Analysis 

## Pre-requisites
You need to satisfy following pre-requisites
 1. Python 2.7
 2. Scikit-Learn 
 3. NLTK
 
## Guide [Ubuntu]

1. Clone the repository by entering following command in terminal
```
git clone https://github.com/TSAwithSSL/TSAwithSSL.git
```

2. Go to repository, and go to src folder and type the following commands
```
cd ..{REPOSITORY}/src
python Visualaizer.py
```
  
3. Then you will get a GUI (which is courrently in development). In GUI we can 
specify the values for label, un label, test and iteration level and do training.
we can also test the tweet once the final model generated.

In addition, GUI is default set to self-training you can modify it to co-training
as well as topic based by replace line 67 in Visualizer.py

```python
self.method = SelfTraining(label, un_label, test, iteration)
```

 1. for doing co-training replace the  above code as

```python
self.method = CoTraining(label , un_label , test, iteration)
```

2. for doing topic-based training replace the  above code as

```python
self.method = TopicOriented(label , un_label , test, iteration)
```
