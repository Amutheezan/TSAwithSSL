from __future__ import division

import csv

import nltk


class MicroBlog:
    def __init__(self):
        self.EMOTICON_FILE = "../resource/emoticon.txt"
        self.UNICODE_EMOTICON_FILE = "../resource/emoticon.csv"
        self._setup_()

    def _setup_(self):
        self.EMOTICON_DICT = self.load_emoticon_dictionary(self.EMOTICON_FILE)
        self.UNICODE_EMOTICON_DICT = self.load_unicode_emoticon_dictionary(self.UNICODE_EMOTICON_FILE)

    def load_emoticon_dictionary(self , filename):
        emo_scores = {'Positive': 0.5 , 'Extremely-Positive': 1.0 , 'Negative': -0.5 , 'Extremely-Negative': -1.0 ,
                      'Neutral': 0.0}
        emo_score_list = {}
        fi = open(filename , "r")
        l = fi.readline()
        while l:
            l = l.replace("\xc2\xa0" , " ")
            li = l.split(" ")
            l2 = li[ :-1 ]
            l2.append(li[ len(li) - 1 ].split("\t")[ 0 ])
            sentiment = li[ len(li) - 1 ].split("\t")[ 1 ][ :-2 ]
            score = emo_scores[ sentiment ]
            l2.append(score)
            for i in range(0 , len(l2) - 1):
                emo_score_list[ l2[ i ] ] = l2[ len(l2) - 1 ]
            l = fi.readline()
        return emo_score_list

    def load_unicode_emoticon_dictionary(self , filename):
        emo_score_list = {}
        with open(filename , "r") as unlabeled_file:
            reader = csv.reader(unlabeled_file)
            for line in reader:
                emo_score_list.update({str(line[ 0 ]): float(line[ 4 ])})
        return emo_score_list

    def emoticon_score(self , tweet):
        s = 0.0
        l = tweet.split(" ")
        nbr = 0
        for i in range(0 , len(l)):
            if l[ i ] in self.EMOTICON_DICT.keys():
                nbr = nbr + 1
                s = s + self.EMOTICON_DICT[ l[ i ] ]
        if nbr != 0:
            s = s / nbr
        return s

    def unicode_emoticon_score(self , tweet):
        s = 0.0
        nbr = 0
        tweet = tweet.split()
        for i in range(len(tweet)):
            old = tweet[ i ]
            new = old.replace("\U000" , "0x")
            if new in self.UNICODE_EMOTICON_DICT.keys():
                nbr = nbr + 1
                s = s + self.UNICODE_EMOTICON_DICT[ new ]
        if nbr != 0:
            s = s / nbr
        return s


class Lexicon:
    def __init__(self , ppros):
        self.ppros = ppros
        self.POS_LEXICON_FILE = "../resource/positive.txt"
        self.NEG_LEXICON_FILE = "../resource/negative.txt"
        self.AFFIN_FILE_96 = "../resource/afinn_96.txt"
        self.AFFIN_FILE_111 = "../resource/afinn_111.txt"
        self.SENTI_140_UNIGRAM_FILE = "../resource/senti140/unigram.txt"
        self.NRC_UNIGRAM_FILE = "../resource/nrc/unigram.txt"
        self.NRC_HASHTAG_FILE = "../resource/nrc/sentimenthash.txt"
        self.BING_LIU_FILE = "../resource/BingLiu.csv"
        self.SENTI_WORD_NET_FILE = "../resource/sentiword_net.txt"
        self._setup_()

    def _setup_(self):
        self.POS_WORDS = self.load_word_dictionary(self.POS_LEXICON_FILE)
        self.NEG_WORDS = self.load_word_dictionary(self.NEG_LEXICON_FILE)
        self.AFFIN_LOAD_96 = self.load_afinn_dictionary(self.AFFIN_FILE_96)
        self.AFFIN_LOAD_111 = self.load_afinn_dictionary(self.AFFIN_FILE_111)
        self.SENTI_140_UNIGRAM_DICT = self.load_generic_dictionary(self.SENTI_140_UNIGRAM_FILE)
        self.NRC_UNIGRAM_DICT = self.load_generic_dictionary(self.NRC_UNIGRAM_FILE)
        self.NRC_HASHTAG_DICT = self.load_generic_dictionary(self.NRC_HASHTAG_FILE)
        self.BING_LIU_DICT = self.load_generic_dictionary(self.BING_LIU_FILE)
        self.SENTI_WORD_NET_DICT = self.load_senti_word_net_dictionary(self.SENTI_WORD_NET_FILE)

    # This for loading dictionaries of NRC,BingLui,senti140 Lexicons
    def load_generic_dictionary(self , file_name):
        lexicon_dict = {}
        f0 = open(file_name , 'r')
        line = f0.readline()
        while line:
            row = line.split("\t")
            line = f0.readline()
            lexicon_dict.update({row[ 0 ]: row[ 1 ]})
        f0.close()
        return lexicon_dict

    # This is for loading dictionaries for Positive,Negative Lexicons
    def load_word_dictionary(self , filename):
        f = open(filename , 'r')
        line = f.readline()
        lines = line.split(",")
        f.close()
        return lines

    # This is for loading AFINN-96 and AFINN-111 dictionaries
    def load_afinn_dictionary(self , filename):
        f = open(filename , 'r')
        afinn = {}
        line = f.readline()
        nbr = 0
        while line:
            try:
                nbr += 1
                l = line[ :-1 ].split('\t')
                afinn[ l[ 0 ] ] = float(l[ 1 ]) / 4
                line = f.readline()
            except ValueError:
                break
        f.close()
        return afinn

    # This is for loading sentiwordnet dictionaries
    def load_senti_word_net_dictionary(self , filename):
        sentiWordnetDict = {}
        tempDictionary = {}
        f0 = open(filename , 'r')
        line = f0.readline()
        line_number = 0
        while (line):
            line_number += 1
            if not ((line.strip()).startswith("#")):
                data = line.split("\t")
                wordTypeMarker = data[ 0 ]
                if len(data) == 6:
                    synsetScore = float(data[ 2 ]) - float(data[ 3 ])
                    synTermsSplit = data[ 4 ].split(" ")
                    for synTermSplit in synTermsSplit:
                        synTermAndRank = synTermSplit.split("#")
                        synTerm = synTermAndRank[ 0 ] + "#" + wordTypeMarker
                        synTermRank = int(synTermAndRank[ 1 ])
                        if not (tempDictionary.has_key(synTerm)):
                            tempDictionary[ str(synTerm) ] = {}
                        tempDictionary[ str(synTerm) ][ str(synTermRank) ] = synsetScore
            line = f0.readline()
        for k1 , v1 in tempDictionary.iteritems():
            score = 0.0
            sum = 0.0
            for k2 , v2 in v1.iteritems():
                score += v2 / float(k2)
                sum += 1.0 / float(k2)
            score /= sum
            sentiWordnetDict[ k1 ] = score
        f0.close()
        return sentiWordnetDict

    def get_ngram_word(self , words , gram):
        ngram_list = [ ]
        for i in range(len(words) + 1 - gram):
            temp = ""
            if not words[ i:i + gram ] is "":
                if gram == 1:
                    temp = words[ i ]
                elif gram == 2:
                    temp = words[ i ] + " " + words[ i + 1 ]
            ngram_list.append(temp)
        return ngram_list

    def get_lexicon_score(self , tweet):
        score = 0
        for w in tweet.split():
            if w in self.POS_LEXICON_FILE:
                score += 1
            if w in self.NEG_LEXICON_FILE:
                score -= 1
            if w.endswith("_NEG"):
                if len(w) > 4:
                    if (w[ 0:len(w) - 4 ]) in self.POS_LEXICON_FILE:
                        score -= 1
                    if (w[ 0:len(w) - 4 ]) in self.NEG_LEXICON_FILE:
                        score += 1
        return score

    def get_afinn_99_score(self , tweet):
        p = 0.0
        nbr = 0
        for w in tweet.split():
            if w in self.AFFIN_LOAD_96.keys():
                nbr += 1
                p += self.AFFIN_LOAD_96[ w ]
            if (w + "_NEG") in self.AFFIN_LOAD_96.keys():
                nbr += 1
                p -= self.AFFIN_LOAD_96[ w ]
        if nbr != 0:
            return p / nbr
        else:
            return 0.0

    def get_afinn_111_score(self , tweet):
        p = 0.0
        nbr = 0
        for w in tweet.split():
            if w in self.AFFIN_LOAD_111.keys():
                nbr += 1
                p += self.AFFIN_LOAD_111[ w ]
            if (w + "_NEG") in self.AFFIN_LOAD_111.keys():
                nbr += 1
                p -= self.AFFIN_LOAD_111[ w ]
        if nbr != 0:
            return p / nbr
        else:
            return 0.0

    def get_senti140_score(self , tweet):
        words = tweet.split()
        unigram_list = self.get_ngram_word(words , 1)
        uni_score = 0.0
        for word in unigram_list:
            if self.SENTI_140_UNIGRAM_DICT.has_key(word):
                uni_score += float(self.SENTI_140_UNIGRAM_DICT.get(word))
            if self.SENTI_140_UNIGRAM_DICT.has_key((word + "_NEG")):
                uni_score -= float(self.SENTI_140_UNIGRAM_DICT.get(word))
        return uni_score

    def get_NRC_score(self , tweet):
        words = tweet.split()
        unigram_list = self.get_ngram_word(words , 1)
        uni_score = 0.0
        hash_score = 0.0
        for word in unigram_list:
            if self.NRC_UNIGRAM_DICT.has_key(word):
                uni_score += float(self.NRC_UNIGRAM_DICT.get(word))
            if self.NRC_UNIGRAM_DICT.has_key((word + "_NEG")):
                uni_score -= float(self.NRC_UNIGRAM_DICT.get(word))
            if list(word)[ 0 ] == '#':
                ar = list(word)
                ar.remove("#")
                word = ''.join(ar)
                if not self.NRC_HASHTAG_DICT.get(word) is None:
                    if self.NRC_HASHTAG_DICT.get(word) == 'positive\n':
                        hash_score += 1.0
                    elif self.NRC_HASHTAG_DICT.get(word) == 'negative\n':
                        hash_score -= 1.0
        return uni_score , hash_score

    def get_senti_word_net_score(self , tweet):
        try:
            nlpos = {'a': [ 'JJ' , 'JJR' , 'JJS' ] ,
                     'n': [ 'NN' , 'NNS' , 'NNP' , 'NNPS' ] ,
                     'v': [ 'VB' , 'VBD' , 'VBG' , 'VBN' , 'VBP' , 'VBZ' , 'IN' ] ,
                     'r': [ 'RB' , 'RBR' , 'RBS' ]}
            text = tweet.split()
            tags = nltk.pos_tag(text)
            tagged_tweets = [ ]
            for i in range(0 , len(tags)):
                if tags[ i ][ 1 ] in nlpos[ 'a' ]:
                    tagged_tweets.append(tags[ i ][ 0 ] + "#a")
                elif tags[ i ][ 1 ] in nlpos[ 'n' ]:
                    tagged_tweets.append(tags[ i ][ 0 ] + "#n")
                elif tags[ i ][ 1 ] in nlpos[ 'v' ]:
                    tagged_tweets.append(tags[ i ][ 0 ] + "#v")
                elif tags[ i ][ 1 ] in nlpos[ 'r' ]:
                    tagged_tweets.append(tags[ i ][ 0 ] + "#r")
            score = 0.0
            for i in range(0 , len(tagged_tweets)):
                if tagged_tweets[ i ] in self.SENTI_WORD_NET_DICT:
                    score += float(self.SENTI_WORD_NET_DICT.get(tagged_tweets[ i ]))
            return score
        except:
            return None

    def get_binliu_score(self , tweet):
        score = 0.0
        for word in tweet.split():
            if self.BING_LIU_DICT.has_key(word):
                if self.BING_LIU_DICT.get(word) == 'positive\n':
                    score += 1
                if self.BING_LIU_DICT.get(word) == 'negative\n':
                    score -= 1
            if self.BING_LIU_DICT.has_key((word + "_NEG")):
                if self.BING_LIU_DICT.get((word + "_NEG")) == 'positive\n':
                    score -= 1
                if self.BING_LIU_DICT.get((word + "_NEG")) == 'negative\n':
                    score += 1
        return score

    def _all_in_lexicon_score_(self , tweet):
        vector = [ ]
        preprocessed_tweet = self.ppros.pre_process_tweet(tweet)
        lexicon_score_gen = self.get_lexicon_score(preprocessed_tweet)
        afinn_score_96 = self.get_afinn_99_score(preprocessed_tweet)
        afinn_score_111 = self.get_afinn_111_score(preprocessed_tweet)
        senti_140_score = self.get_senti140_score(preprocessed_tweet)
        n_r_c_score = self.get_NRC_score(preprocessed_tweet)
        binliu_score = self.get_senti_word_net_score(preprocessed_tweet)
        sentiword_score = self.get_binliu_score(preprocessed_tweet)
        vector.append(afinn_score_96)
        vector.append(afinn_score_111)
        vector.append(lexicon_score_gen)
        vector.append(senti_140_score)
        vector.extend(n_r_c_score)
        vector.append(binliu_score)
        vector.append(sentiword_score)
        return vector


class WritingStyle:
    def capitalized_words_in_tweet(self , tweet):
        count = 0
        if len(tweet) != 0:
            for w in tweet.split():
                if w.isupper():
                    if len(w) > 1:
                        count = count + 1
        return count

    def exclamation_count(self , tweet):
        tweet_words = tweet.split()
        count = 0
        for word in tweet_words:
            if word.count("!") > 2:
                count += 1
        return count

    def question_mark_count(self , tweet):
        tweet_words = tweet.split()
        count = 0
        for word in tweet_words:
            if word.count("?") > 2:
                count += 1
        return count

    def capital_count_in_a_word(self , tweet):
        count = 0
        if len(tweet) != 0:
            for c in tweet:
                if str(c).isupper():
                    count = count + 1
        return count

    def surround_by_signs(self , tweet):
        highlight = [ '"' , "'" , "*" ]
        count = 0
        if len(tweet) != 0:
            for c in tweet:
                if c[ 0 ] == c[ len(c) - 1 ] and c[ 0 ] in highlight:
                    count = count + 1
        return count

    def writing_style_vector(self, tweet):
        cap_word = self.capitalized_words_in_tweet(tweet)
        exc_count = self.exclamation_count(tweet)
        que_count = self.question_mark_count(tweet)
        cap_count = self.capital_count_in_a_word(tweet)
        surr_count = self.surround_by_signs(tweet)
        return [ cap_word , exc_count , que_count , cap_count , surr_count ]


class NGram:
    def __init__(self , commons, ppros):
        self.commons = commons
        self.ppros = ppros

    def create_dict(self , words , gram):
        """
        This is to obtain the create_dict of word of particular line
        :param words: 
        :param gram: 
        :return: give a create_dict of word(s) with proper format based generate_n_gram_dict values such as 1,2,3
        """
        temp_dict = {}
        for i in range(len(words) - gram):
            if not words[ i:i + gram ] is "":
                temp = ""
                if gram == 1:
                    temp = words[ i ]
                elif gram == 2:
                    temp = words[ i ] + " " + words[ i + 1 ]
                elif gram == 3:
                    temp = words[ i ] + " " + words[ i + 1 ] + " " + words[ i + 2 ]
                local_temp_value = temp_dict.get(temp)
                if local_temp_value is None:
                    temp_dict.update({temp: 1})
                else:
                    temp_dict.update({temp: local_temp_value + 1})
        return temp_dict

    def generate_n_gram_dict(self , file_dict , gram):
        """
        this will return n-gram set for uni-gram,bi-gram and tri-gram, with frequency calculated
        for normal text and POS-tagged.
        :param file_dict:
        :param gram:
        :return: frequency dictionaries
        """
        word_freq_dict = {}
        postag_freq_dict = {}
        keys = file_dict.keys()
        for line_key in keys:
            try:
                line = file_dict.get(line_key)
                words = line.split()
                word_dict = self.create_dict(words , gram)
                word_freq_dict , is_success = self.commons.dict_update(word_freq_dict , word_dict)
                temp_postags = self.ppros.pos_tag_string(line).split()
                if temp_postags != "":
                    postags = temp_postags
                    postag_dict = self.create_dict(postags , gram)
                    postag_freq_dict , is_success = self.commons.dict_update(postag_freq_dict , postag_dict)
            except IndexError:
                print "Error"
        return word_freq_dict , postag_freq_dict

    def score(self , tweet , p , n , ne , ngram):
        """
        This will find individual score of each word with respect to its polarity
        :param tweet: 
        :param p: 
        :param n: 
        :param ne: 
        :param ngram: 
        :return: return positive, negative, and neutral score
        """
        pos = 0
        neg = 0
        neu = 0
        dictof_grams = {}
        tweet_list = tweet.split()
        dictof_grams.update(self.create_dict(tweet_list , ngram))
        for element in dictof_grams.keys():
            posCount = float(self.get_count(element , p))
            negCount = float(self.get_count(element , n))
            neuCount = float(self.get_count(element , ne))
            totalCount = posCount + negCount + neuCount
            if totalCount != 0:
                pos += posCount / totalCount
                neg += negCount / totalCount
                neu += neuCount / totalCount
        return [ pos , neg , neu ]

    def get_count(self , gram , pol):
        """
        This will count the positive,negative, and neutral count based on relevant dictionary present
        :param gram: 
        :param pol: 
        :return: return the availability of particular generate_n_gram_dict
        """
        count = 0.0
        try:
            temp = float(pol.get(gram))
            if temp > 0.0:
                count = temp
        except:
            TypeError
        return count
