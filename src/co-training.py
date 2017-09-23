import csv
import time

import _config_constants_ as cons
import _generic_commons_ as commons
import _load_model_test_iterate_ as lmti


def co_training():
    final_file = open('../dataset/analysed/' + lmti.get_file_prefix() + str(time.time()) + 'result.csv' , 'w+')
    csv_result = csv.writer(final_file)
    csv_result.writerow(cons.CSV_HEADER)

    time_list = [ time.time() ]

    result = lmti.initial_run()
    print result
    csv_result.writerow(result)

    while lmti.ds.STABILITY_BREAK > 0:
        if lmti.ds.CURRENT_ITERATION == 0:
            is_co_training = False
        else:
            is_co_training = True
        result = lmti.co_training_run(is_co_training)
        print result
        csv_result.writerow(result)

    final_file.close()
    time_list.append(time.time())
    print commons.temp_difference_cal(time_list)


co_training()
