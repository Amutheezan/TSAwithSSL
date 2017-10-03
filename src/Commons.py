import math


class Commons:

    def __init__(self, config):
        self.config = config

    def temp_difference_cal(self,time_list):
        """
        This function is used when a set of time values are added and difference between last two are obtained
        :param time_list:
        :return: difference
        """
        if len(time_list) > 1:
            final = float(time_list[len(time_list) - 1])
            initial = float(time_list[len(time_list) - 2])
            difference = final - initial
        else:
            difference = -1.0
        return difference

    def dict_update(self,original, temp):
        """
        This will update original dictionary key, and values by comparing with temp values
        :param original:
        :param temp:
        :return: original updated dictionary and a success statement
        """
        is_success = False
        result = {}
        original_temp = original.copy()
        for key in temp.keys():
            global_key_value = original_temp.get(key)
            local_key_value = temp.get(key)
            if key not in original_temp.keys():
                result.update({key: local_key_value})
            else:
                result.update({key: local_key_value + global_key_value})
                del original_temp[key]
        result.update(original_temp)
        return result, is_success

    def get_divided_value(self,numerator, denominator):
        if denominator == 0:
            return 0.0
        else:
            result = numerator / (denominator * 1.0)
            return round(result, 4)

    def get_entropy(self,proba):
        """
        calculate entropy based formula -p(x)log(x)
        :param proba: 
        :return: 
        """
        if proba == 0:
            return 0.0
        else:
            return round(-1.0 * (proba) * math.log(proba, 2), 2)

    def first_next_max(self,input_list):
        first = 0.0
        next = 0.0
        for element in input_list:
            if element > first:
                next = first
                first = element
            elif element > next:
                next = element
        return first , next

    def get_labels(self,p , prob):
        l = self.config.UNLABELED
        if prob[ 0 ] == p:
            l = self.config.LABEL_NEGATIVE
        if prob[ 1 ] == p:
            l = self.config.LABEL_NEUTRAL
        if prob[ 2 ] == p:
            l = self.config.LABEL_POSITIVE
        return l

    def get_sum_proba(self,first , second):
        result = [ ]
        for i in range(len(first)):
            result.append(0.5 * (first[ i ] + second[ i ]))
        return result

    def get_file_prefix(self):
        return "{0}_{1}_{2}_". \
            format(
            str(self.config.LABEL_LIMIT) ,
            str(self.config.TEST_LIMIT) , str(self.config.DEFAULT_CLASSIFIER)
        )

    def get_values(self,actual, predict):
        TP = 0
        TN = 0
        TNeu = 0
        FP_N = 0
        FP_Neu = 0
        FN_P = 0
        FN_Neu = 0
        FNeu_P = 0
        FNeu_N = 0
        for i in range(len(actual)):
            a = actual[i]
            p = predict[i]
            if a == p:
                if a == self.config.LABEL_POSITIVE:
                    TP +=1
                if a == self.config.LABEL_NEUTRAL:
                    TNeu +=1
                if a == self.config.LABEL_NEGATIVE:
                    TN +=1
            if a != p:
                if a == self.config.LABEL_POSITIVE:
                    if p == self.config.LABEL_NEGATIVE:
                        FN_P +=1
                    if p == self.config.LABEL_NEUTRAL:
                        FNeu_P +=1
                if a == self.config.LABEL_NEGATIVE:
                    if p == self.config.LABEL_POSITIVE:
                        FP_N += 1
                    if p == self.config.LABEL_NEUTRAL:
                        FNeu_N += 1
                if a == self.config.LABEL_NEUTRAL:
                    if p == self.config.LABEL_POSITIVE:
                        FP_Neu += 1
                    if p == self.config.LABEL_NEGATIVE:
                        FN_Neu += 1

        accuracy = self.get_divided_value(TP+TN+TNeu, FP_N+FP_Neu+FN_P+FN_Neu+FNeu_P+FNeu_N)
        pre_pos = self.get_divided_value(TP , TP + FP_Neu + FP_N)
        pre_neg = self.get_divided_value(TN , TN + FN_Neu + FN_P)
        re_pos = self.get_divided_value(TP,TP+FNeu_P+FN_P)
        re_neg = self.get_divided_value(TN,TN+FNeu_N+FP_N)
        f1_pos = 2 * self.get_divided_value(pre_pos * re_pos , pre_pos + re_pos)
        f1_neg = 2 * self.get_divided_value(pre_neg * re_neg , pre_neg + re_neg)
        return accuracy,pre_pos,pre_neg,re_pos,re_neg,f1_pos,f1_neg, 0.5 * (f1_neg + f1_pos)