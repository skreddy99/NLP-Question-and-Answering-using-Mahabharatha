import keras
from keras.layers import LSTM, Dense, Activation, Bidirectional, Dropout
from keras.optimizers import RMSprop, Adam, Adagrad, SGD
import datautils as dt
from keras.models import Sequential
from keras import regularizers
import os
import numpy as np
import pickle
from itertools import islice
from keras.models import load_model

# Create inputs for prediction

#Here is teh directory path; create a subdiretory called 'predict' at the same level of 'data'.
# in other words, create a subdirectory at parent directory



def create_inputs_for_prediction(dirpath):
    train_pctx_len = []
    train_pq_len = []
    train_px_list = []
    columnsize = 0
    train_px_vector = []

    train_pctxfiles = [name for name in os.listdir(dirpath) if name.endswith('_ctx') and name.startswith('train_')]
    train_pqueryfiles = [name for name in os.listdir(dirpath) if name.endswith('_q') and name.startswith('train_')]

    print('finding the max and min length of context and query files....')
    # find the maxlength of number of words in context and query files. It will be used to create the input shape.
    for filename in train_pctxfiles:
        filepath = os.path.join(dirpath, filename)
        tmp = pickle.load(open(filepath, 'rb'))
        train_pctx_len.append(len(tmp))
        columnsize = len(tmp[0])

    for filename in train_pqueryfiles:
        filepath = os.path.join(dirpath, filename)
        train_pq_len.append(len(pickle.load(open(filepath, 'rb'))))


    max_pctx_len = max(train_pctx_len)
    max_pctx_len = 250
    max_pq_len = max(train_pq_len)

    print('max train context length is %d and min train context length is %d' % (max_pctx_len, min(train_pctx_len)))
    print('max train query length is %d and min train query length is %d' % (max_pq_len, min(train_pq_len)))


    train_pctxfiles = [name for name in os.listdir(dirpath)
                      if name.endswith('_ctx') and name.startswith('train_')]
    train_pqueryfiles = [name for name in os.listdir(dirpath)
                        if name.endswith('_q') and name.startswith('train_')]

    max_pctx_len = max(train_pctx_len)
    max_pq_len = max(train_pq_len)
    min_pctx_len = min(train_pctx_len)
    min_pq_len = min(train_pq_len)
    print('Before restriction, max train context length is %d and '
          'min train context length is %d' % (max_pctx_len, min_pctx_len))
    print('Before restriction, max train query length is %d and '
          'min train query length is %d' % (max_pq_len, min_pq_len))


    max_pctx_len = 350
    max_pq_len = 40
    min_pctx_len = 50
    min_pq_len = 0
    print('\n\n')
    print('Max ctx len is ', max_pctx_len)
    print('Max q len is ', max_pq_len)
    print('Min ctx len is ', min_pctx_len)
    print('Min q len is ', min_pq_len)
    print('\n\n')



    count = 0
    n_train_contexts = 500
    start = 0
    for i in range(len(train_pctxfiles)//n_train_contexts):
        end = start+n_train_contexts  # number of contexts to be stored in a single input pickle
        train_pctx_len = []
        train_pq_len = []
        train_px_list = []
        n_temp = 0
        while (start < end):
            filename = train_pctxfiles[start]

            pctx_filepath = os.path.join(dirpath, filename)
            tmp_c = pickle.load(open(pctx_filepath, 'rb'))
            pquery_filepath = os.path.join(dirpath, filename[:-3] + 'q')
            tmp_q = pickle.load(open(pquery_filepath, 'rb'))

            if (len(tmp_c) <= max_pctx_len) and \
                    (len(tmp_q) <= max_pq_len) and \
                    (len(tmp_c) >= min_pctx_len) and \
                    (len(tmp_q) >= min_pq_len):

                columnsize = len(tmp_c[0])
                train_pctx_len.append(len(tmp_c))
                train_pq_len.append(len(tmp_q))
                c = np.array(tmp_c)
                q = np.array(tmp_q)
                cq = np.concatenate((c, q), axis=0)
                if ((max_pctx_len+max_pq_len) - (len(tmp_c)+len(tmp_q)) > 0):
                    pad_zeros = max_pctx_len + max_pq_len - (len(tmp_c)+len(tmp_q))
                    pad = np.zeros((pad_zeros, columnsize))
                    cq = np.concatenate((cq, pad), axis=0)
                train_px_list.append(cq)
            start+=1
            print(start)
        train_px_vector = np.array(train_px_list)
        train_px_vector = train_px_vector.reshape(len(train_pctx_len),
                                                max_pctx_len + max_pq_len, columnsize)
        print('\n\n' + 'Length of each Training Context is....')
        print(train_pctx_len)
        print('Number of contexts read is ', len(train_pctx_len))
        print('\n\n' + 'Length of each Training Query is....')
        print(train_pq_len)
        print('Number of queries read is ', len(train_pq_len))
        print('\n\n'+'Train data X-vector shape is %s' % str(train_px_vector.shape))
        pickle.dump(train_px_vector, open('train_px_vector.pickle'+str(count), 'a+b'))
        count+=1
        print('\n\n' + 'Pickle number ', count, ' is stored')

        print('\n\n'+'Number of train pickles stored are ', count)






if __name__ == '__main__':
#    dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    pdirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'predict'))
    create_inputs(pdirpath)
    '''
    print('X_train_shape is')
    print(X_train.shape)
    print('y_train_shape is')
    print(y_train.shape)
    print('X_test_shape is')
    print(X_test.shape)
    print('y_test_shape is')
    print(y_test.shape)


    lstm_model0()
    lstm_model1()
    lstm_model2()
    lstm_model3()
    bi_lstm_model0()
    bi_lstm_model1()
    bi_lstm_model2()
    bi_lstm_model3()
    bi_lstm_model4()

    '''







