import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag


class PreProcess:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = []
        self.slangs = {}
        self._setup_()

    def _setup_(self):
        self.load_stop_word_list()
        self.load_internet_slangs_dict()

    def load_stop_word_list(self):
        fp = open("../resource/stopWords.txt" , 'r')
        self.stop_words = [ 'at_user' , 'url' ]
        line = fp.readline()
        while line:
            word = line.strip()
            self.stop_words.append(word)
            line = fp.readline()
        fp.close()
        return

    def remove_stop_words(self , tweet):
        result = ''
        for w in tweet:
            if w[ -4: ] == "_NEG":
                if w[ :-4 ] not in self.stop_words:
                    result = result + w + ' '
            else:
                if w not in self.stop_words:
                    result = result + w + ' '
        return result

    def negate(self , tweets):
        fn = open("../resource/negation.txt" , "r")
        line = fn.readline()
        negation_list = [ ]
        while line:
            negation_list.append(line.split(None , 1)[ 0 ])
            line = fn.readline()
        fn.close()
        punctuation_marks = [ "." , ":" , ";" , "!" , "?" ]
        break_words = [ "but" ]

        for i in range(len(tweets)):
            if tweets[ i ] in negation_list:
                j = i + 1
                while j < len(tweets):
                    if tweets[ j ][ -1 ] not in (punctuation_marks and break_words):
                        tweets[ j ] = tweets[ j ] + "_NEG"
                        j = j + 1
                    elif tweets[ j ][ -1 ] not in (punctuation_marks and break_words):
                        tweets[ j ] = tweets[ j ][ -1 ] + "_NEG"
                    else:
                        break
                i = j
        return tweets

    def load_internet_slangs_dict(self):
        fi = open('../resource/internetSlangs.txt' , 'r')
        line = fi.readline()
        while line:
            l = line.split(r',%,')
            if len(l) == 2:
                self.slangs[ l[ 0 ] ] = l[ 1 ][ :-2 ]
            line = fi.readline()
        fi.close()
        return

    def replace_slangs(self , tweet):
        result = ''
        words = tweet.split()
        for w in words:
            if w in self.slangs.keys():
                result = result + self.slangs[w] + " "
            else:
                result = result + w + " "
        return result

    def replace_two_or_more(self , s):
        pattern = re.compile(r"(.)\1{1,}" , re.DOTALL)
        return pattern.sub(r"\1\1" , s)

    def pos_tag_string(self , tweet):
        adjective_list = [ "JJ" , "JJR" , "JJS" ]
        verb_list = [ "VB" , "VBD" , "VBG" , "VBN" , "VBP" , "VBZ" ]
        adverb_list = [ "RB" , "RBR" , "RBS" , "WRB" ]
        noun_list = [ 'NN' , 'NNS' , 'NNP' , 'NNPS' ]
        tag_tweet = ""
        tweet_words = ""
        tweet = tweet.split()
        for words in tweet:
            tweet_words += words + " "
        tagged_tweet = (nltk.pos_tag(tweet_words.replace("_NEG" , "").split()))
        for i in range(len(tagged_tweet)):
            if tagged_tweet[ i ][ 1 ] in adjective_list:
                tag_tweet += tagged_tweet[ i ][ 0 ] + "|" + "ADJ" + " "
            elif tagged_tweet[ i ][ 1 ] in verb_list:
                tag_tweet += tagged_tweet[ i ][ 0 ] + "|" + "VER" + " "
            elif tagged_tweet[ i ][ 1 ] in adverb_list:
                tag_tweet += tagged_tweet[ i ][ 0 ] + "|" + "ADV" + " "
            elif tagged_tweet[ i ][ 1 ] in noun_list:
                tag_tweet += tagged_tweet[ i ][ 0 ] + "|" + "NOU" + " "
        return tag_tweet

    def pre_process_tweet(self , tweet):
        tweet = tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))' , 'url' , tweet)
        tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+))' , 'url' , tweet)
        tweet = re.sub('@[^\s]+' , 'at_user' , tweet)
        tweet = re.sub('[\s]+' , ' ' , tweet)
        tweet = tweet.strip('\'"')
        processed_tweet = self.replace_two_or_more(tweet)
        words = self.replace_slangs(processed_tweet).split()
        negated_tweets = self.negate(words)
        preprocessed_tweet = self.remove_stop_words(negated_tweets)
        punctuation_removed_tweets = re.sub('[^a-zA-Z_]+' , ' ' , preprocessed_tweet)
        final_processed_tweets = punctuation_removed_tweets.replace(" _NEG" , "_NEG")
        return final_processed_tweets
