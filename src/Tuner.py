import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn import preprocessing as pr , svm
from SystemStore import *
from Features import *
from PreProcess import PreProcess

warnings.filterwarnings('ignore')


class Tuner:
    def __init__(self , label , train_type):
        self.cons = Constants()
        self.train_contents = self.cons.TRAIN_SET.get(train_type)
        self.pre_pros = PreProcess()
        self.mb = MicroBlog()
        self.ws = WritingStyle()
        self.ds = DataStore(self.cons)
        self.commons = Commons(self.cons)
        self.lexicon = Lexicon(self.pre_pros , self.cons)
        self.n_gram = NGram(self.commons , self.cons , self.pre_pros)
        self.LABEL_LIMIT = min(self.train_contents.get("size") , label)
        self.POS_COUNT_LIMIT = int(self.LABEL_LIMIT * self.train_contents.get("pos_ratio"))
        self.NEG_COUNT_LIMIT = int(self.LABEL_LIMIT * self.train_contents.get("neg_ratio"))
        self.NEU_COUNT_LIMIT = int(self.LABEL_LIMIT * self.train_contents.get("neu_ratio"))
        self.N_GRAM_LIMIT = 1
        self.NO_OF_MODELS = 1
        self.TRAINING_TYPE = self.cons.SELF_TRAINING_TYPE

    def load_tune_dictionary(self):
        temp_train_dict = {}
        temp_tune_dict = {}
        with open(self.train_contents.get("train_file") , 'r') as main_dataset:
            main = csv.reader(main_dataset)
            pos_count = 1
            neg_count = 1
            neu_count = 1
            count = 1
            for line in main:
                count += 1
                if line[ 0 ] == self.cons.NAME_POSITIVE and pos_count <= self.POS_COUNT_LIMIT:
                    temp_train_dict.update(
                        {str(count): [ str(line[ 1 ]) , self.cons.LABEL_POSITIVE , 1 , 1 , self.cons.NO_TOPIC , 0 ]})
                    pos_count += 1

                if line[ 0 ] == self.cons.NAME_NEGATIVE and neg_count <= self.NEG_COUNT_LIMIT:
                    temp_train_dict.update(
                        {str(count): [ str(line[ 1 ]) , self.cons.LABEL_NEGATIVE , 1 , 1 , self.cons.NO_TOPIC , 0 ]})
                    neg_count += 1

                if line[ 0 ] == self.cons.NAME_NEUTRAL and neu_count <= self.NEU_COUNT_LIMIT:
                    temp_train_dict.update(
                        {str(count): [ str(line[ 1 ]) , self.cons.LABEL_NEUTRAL , 1 , 1 , self.cons.NO_TOPIC , 0 ]})
                    neu_count += 1
        self.ds.TRAIN_DICT = temp_train_dict
        self.ds.POS_INITIAL = pos_count
        self.ds.NEG_INITIAL = neg_count
        self.ds.NEU_INITIAL = neu_count
        self.ds.POS_SIZE = pos_count
        self.ds.NEG_SIZE = neg_count
        self.ds.NEU_SIZE = neu_count

        with open(self.cons.FILE_TUNE , 'r') as tune_dataset:
            tune = csv.reader(tune_dataset)
            pos_count = 1
            neg_count = 1
            neu_count = 1
            count = 1
            for line in tune:
                count += 1
                if line[ 0 ] == self.cons.NAME_POSITIVE and pos_count <= self.POS_COUNT_LIMIT:
                    temp_tune_dict.update(
                        {str(count): [ str(line[ 1 ]) , self.cons.LABEL_POSITIVE , 1 , 1 , self.cons.NO_TOPIC , 0 ]})
                    pos_count += 1

                if line[ 0 ] == self.cons.NAME_NEGATIVE and neg_count <= self.NEG_COUNT_LIMIT:
                    temp_tune_dict.update(
                        {str(count): [ str(line[ 1 ]) , self.cons.LABEL_NEGATIVE , 1 , 1 , self.cons.NO_TOPIC , 0 ]})
                    neg_count += 1

                if line[ 0 ] == self.cons.NAME_NEUTRAL and neu_count <= self.NEU_COUNT_LIMIT:
                    temp_tune_dict.update(
                        {str(count): [ str(line[ 1 ]) , self.cons.LABEL_NEUTRAL , 1 , 1 , self.cons.NO_TOPIC , 0 ]})
                    neu_count += 1

        self.ds.TUNE_DICT = temp_tune_dict
        return

    def load_matrix_sub(self , process_dict , mode , label):
        limit = self.LABEL_LIMIT
        if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
            vectors = {}
            labels = {}
        else:
            vectors = [ ]
            labels = [ ]
        if limit != 0:
            keys = process_dict.keys()
            if len(keys) > 0:
                for key in keys:
                    if label == process_dict.get(key)[ 1 ]:
                        line = process_dict.get(key)[ 0 ]
                        z = self.map_tweet(line , mode)
                        if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
                            vectors.update({key: z})
                            labels.update({key: label})
                        else:
                            vectors.append(z)
                            labels.append(label)
        return vectors , labels

    def map_tweet_n_gram_values(self , tweet):
        vector = [ ]

        preprocessed_tweet = self.pre_pros.pre_process_tweet(tweet)
        pos_tag_tweet = self.pre_pros.pos_tag_string(preprocessed_tweet)
        for n in range(1 , self.N_GRAM_LIMIT + 1 , 1):
            pos , neg , neu = self.ds._get_n_gram_(n , False)
            n_gram_score = self.n_gram.score(preprocessed_tweet , pos , neg , neu , n)
            pos , neg , neu = self.ds._get_n_gram_(n , True)
            post_n_gram_score = self.n_gram.score(pos_tag_tweet , pos , neg , neu , n)
            vector.extend(n_gram_score)
            vector.extend(post_n_gram_score)
        return vector

    def map_tweet_feature_values(self , tweet):
        vector = [ ]
        vector.append(self.mb.emoticon_score(tweet))
        vector.append(self.mb.unicode_emoticon_score(tweet))
        vector.extend(self.ws.writing_style_vector(tweet))
        vector.extend(self.lexicon._all_in_lexicon_score_(tweet))
        return vector

    def generate_vectors_and_labels(self):
        for n in range(1 , self.N_GRAM_LIMIT + 1 , 1):
            pos , pos_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_POSITIVE , n)
            neg , neg_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_NEGATIVE , n)
            neu , neu_p = self.n_gram.generate_n_gram_dict(self.ds.TRAIN_DICT , self.cons.LABEL_NEUTRAL , n)

            self.ds._update_uni_gram_(pos , neg , neu , n , False)
            self.ds._update_uni_gram_(pos_p , neg_p , neu_p , n , True)

        for mode in range(self.NO_OF_MODELS):
            if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
                vectors_dict = {}
                labels_dict = {}
            else:
                vectors_list = [ ]
                labels_list = [ ]
            for label in self.cons.LABEL_TYPES:
                vec , lab = self.load_matrix_sub(self.ds.TRAIN_DICT , mode , label)
                if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
                    vectors_dict.update(vec)
                    labels_dict.update(lab)
                else:
                    vectors_list += vec
                    labels_list += lab
            if self.TRAINING_TYPE == self.cons.TOPIC_BASED_TRAINING_TYPE:
                self.ds._dump_vectors_labels_(vectors_dict , labels_dict , mode)
            else:
                self.ds._dump_vectors_labels_(vectors_list , labels_list , mode)
        return

    def get_class_weight(self):
        pos = self.ds.POS_SIZE
        neg = self.ds.NEG_SIZE
        neu = self.ds.NEU_SIZE
        weights = dict()
        weights[ self.cons.LABEL_POSITIVE ] = (1.0 * neu) / pos
        weights[ self.cons.LABEL_NEGATIVE ] = (1.0 * neu) / neg
        weights[ self.cons.LABEL_NEUTRAL ] = 1.0
        return weights

    def tune_run(self):
        self.load_tune_dictionary()
        print "load tune dictionary"
        self.generate_vectors_and_labels()
        print "generated vectors"
        self.do_tuning()

    def do_tuning(self):
        vectors = self.ds._get_vectors_(0)
        labels = self.ds._get_labels_(0)
        for label in self.cons.LABEL_TYPES:
            vec , lab = self.load_matrix_sub(self.ds.TUNE_DICT , 0 , label)
            vectors += vec
            labels += lab
        vectors_scaled = pr.scale(vectors)
        vectors_normalized = pr.normalize(vectors_scaled , norm='l2')
        vectors = vectors_normalized
        vectors = vectors.tolist()
        test_fold = [ ]
        for i in range(1 , len(vectors)):
            if i < 20631:
                test_fold.append(-1)
            else:
                test_fold.append(0)
        svr = svm.SVC(class_weight=self.get_class_weight())
        ps = PredefinedSplit(test_fold=test_fold)
        kernel_list = [ self.cons.KERNEL_RBF ]

        initial_ratio = 0.01
        c_start = 1
        gamma_start = 1
        total_range = 128
        c_end = c_start + total_range
        gamma_end = gamma_start + total_range
        c_latest = 0.01
        gamma_latest = 0.01
        score_latest = 0.00
        while True:
            score_previous = score_latest
            c_previous = c_latest
            gamma_previous = gamma_latest
            c_step = (c_end - c_start) / 2
            gamma_step = (gamma_end - gamma_start) / 2
            c_range = [ initial_ratio * i for i in range(c_start , c_end + 1 , c_step) ]
            gamma_range = [ initial_ratio * i for i in range(gamma_start , gamma_end + 1 , gamma_step) ]
            print "range of c" , c_range
            print "range of gamma" , gamma_range
            parameters = {'kernel': kernel_list , 'C': c_range , 'gamma': gamma_range}
            grid = GridSearchCV(svr , parameters , scoring='f1_weighted' , n_jobs=-1 , cv=ps)
            tunes_model = grid.fit(vectors , labels)
            results = tunes_model.best_params_
            score_latest = tunes_model.best_score_
            c_latest = results[ 'C' ]
            gamma_latest = results[ 'gamma' ]
            if c_latest == c_previous and gamma_latest == gamma_previous \
                    and (score_latest <= score_previous or total_range == 2):
                break
            else:
                c_start = int(max(1 , c_latest * 100 - total_range / 4))
                c_end = int(c_latest * 100 + total_range / 4)
                gamma_start = int(max(1 , gamma_latest * 100 - total_range / 4))
                gamma_end = int(gamma_latest * 100 + total_range / 4)
                total_range = total_range / 2
        print c_latest , gamma_latest

    def map_tweet(self , tweet , mode):
        vector = []
        if not mode:
            vector.extend(self.map_tweet_feature_values(tweet))
            vector.extend(self.map_tweet_n_gram_values(tweet))
        return vector

c = Constants()
t = Tuner(20631, c.TRAIN_2017)
t.tune_run()