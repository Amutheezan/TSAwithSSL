import math


def temp_difference_cal(time_list):
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


def dict_update(original, temp):
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


def get_divided_value(numerator, denominator):
    if denominator == 0:
        return 0.0
    else:
        result = numerator / (denominator * 1.0)
        return round(result, 4)


def get_entropy(proba):
    """
    calculate entropy based formula -p(x)log(x)
    :param proba: 
    :return: 
    """
    if proba == 0:
        return 0.0
    else:
        return round(-1.0 * (proba) * math.log(proba, 2), 2)


def find_first_second_max(input_list):
    first_max = 0.0
    second_max = 0.0
    for element in input_list:
        if element > first_max:
            second_max = first_max
            first_max = element
    return first_max,second_max