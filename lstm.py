import keras
from keras.layers import LSTM, Dense, Activation, Bidirectional, Dropout
from keras.optimizers import RMSprop, Adam, Adagrad, SGD
import datautils as dt
from keras.models import Sequential
from keras import regularizers
import os, sys
import numpy as np
import pickle
from itertools import islice
from keras.models import load_model

def create_inputs(data_dir):
    """

    :param dirpath:
    :return:
    """
    train_ctx_len = []
    train_q_len = []
    test_ctx_len = []
    test_q_len = []
    test_a_list = []
    train_x_list = []
    test_x_list = []
    columnsize = 0
    train_x_vector = []
    train_y_vector = []
    test_x_vector = []
    test_y_vector = []



    '''

    train_ctxfiles = [name for name in os.listdir(dirpath) if name.endswith('_ctx') and name.startswith('train_')]
    train_queryfiles = [name for name in os.listdir(dirpath) if name.endswith('_q') and name.startswith('train_')]
    test_ctxfiles = [name for name in os.listdir(dirpath) if name.endswith('_ctx') and name.startswith('test_')]

    print('finding the max and min length of context and query files....')
    # find the maxlength of number of words in context and query files. It will be used to create the input shape.
    for filename in train_ctxfiles:
        filepath = os.path.join(dirpath, filename)
        tmp = pickle.load(open(filepath, 'rb'))
        train_ctx_len.append(len(tmp))
        columnsize = len(tmp[0])

    for filename in train_queryfiles:
        filepath = os.path.join(dirpath, filename)
        train_q_len.append(len(pickle.load(open(filepath, 'rb'))))


#    max_ctx_len = max(train_ctx_len)
    max_ctx_len = 250
    max_q_len = max(train_q_len)

    print('max train context length is %d and min train context length is %d' % (max_ctx_len, min(train_ctx_len)))
    print('max train query length is %d and min train query length is %d' % (max_q_len, min(train_q_len)))


    '''


    train_ctxfiles = [name for name in os.listdir(data_dir)
                      if name.endswith('_ctx') and name.startswith('train_')]
#    train_queryfiles = [name for name in data_dir
#                        if name.endswith('_q') and name.startswith('train_')]
    test_ctxfiles = [name for name in os.listdir(data_dir)
                     if name.endswith('_ctx') and name.startswith('test_')]
    '''
    max_ctx_len = max(train_ctx_len)
    max_q_len = max(train_q_len)
    min_ctx_len = min(train_ctx_len)
    min_q_len = min(train_q_len)
    print('Before restriction, max train context length is %d and '
          'min train context length is %d' % (max_ctx_len, min_ctx_len))
    print('Before restriction, max train query length is %d and '
          'min train query length is %d' % (max_q_len, min_q_len))
    '''

    max_ctx_len = 350
    max_q_len = 40
    min_ctx_len = 50
    min_q_len = 0
    print('\n\n')
    print('Max ctx len is ', max_ctx_len)
    print('Max q len is ', max_q_len)
    print('Min ctx len is ', min_ctx_len)
    print('Min q len is ', min_q_len)
    print('\n\n')


# Check if all the 4 input & label for training and input & label testing pickles exist in the pickle directory;
# If any one pickle is missing, create al teh four pickles new

    file_exists = 0
    if os.path.exists(os.path.join(pickle_dir, 'train_x_vector.pickle1')):
        file_exists +=1
#        train_x_vector = np.array(pickle.load(open(os.path.join(pickle_dir, 'train_x_vector.pickle1'), 'rb'))

    if os.path.exists(os.path.join(pickle_dir, 'train_y_vector.pickle1')):
        file_exists +=1
#        train_y_vector = pickle.load(open(os.path.join(pickle_dir, 'train_y_vector.pickle1'), 'rb'))

    if os.path.exists(os.path.join(pickle_dir, 'test_x_vector.pickle1')):
        file_exists +=1
#        test_x_vector = pickle.load(open(os.path.join(pickle_dir, 'test_x_vector.pickle1'), 'rb'))


    if os.path.exists(os.path.join(pickle_dir, 'test_y_vector.pickle1')):
        file_exists +=1
#        test_y_vector = pickle.load(open(os.path.join(pickle_dir, 'test_y_vector.pickle1'), 'rb'))

    if file_exists < 4: # Checking if any of the training or test input & label pickles are missing
        count = 0
        n_train_contexts = 500 # set the limit on the number of contexts to store in a pickle; a large pickle my impact the memory
        start = 0
        for i in range(len(train_ctxfiles)//n_train_contexts):
            end = start+n_train_contexts  # number of contexts to be stored in each iteration
            train_a_list = []
            train_ctx_len = []
            train_q_len = []
            train_x_list = []
            n_temp = 0
            while (start < end):
                filename = train_ctxfiles[start]

                ctx_filepath = os.path.join(data_dir, filename)
                tmp_c = pickle.load(open(ctx_filepath, 'rb'))
                query_filepath = os.path.join(data_dir, filename[:-3] + 'q')
                tmp_q = pickle.load(open(query_filepath, 'rb'))

                if (len(tmp_c) <= max_ctx_len) and \
                        (len(tmp_q) <= max_q_len) and \
                        (len(tmp_c) >= min_ctx_len) and \
                        (len(tmp_q) >= min_q_len):

                    columnsize = len(tmp_c[0])
                    train_ctx_len.append(len(tmp_c))
                    train_q_len.append(len(tmp_q))
                    c = np.array(tmp_c)
                    q = np.array(tmp_q)
                    cq = np.concatenate((c, q), axis=0)
                    if ((max_ctx_len+max_q_len) - (len(tmp_c)+len(tmp_q)) > 0):
                        pad_zeros = max_ctx_len + max_q_len - (len(tmp_c)+len(tmp_q))
                        pad = np.zeros((pad_zeros, columnsize))
                        cq = np.concatenate((cq, pad), axis=0)
                    train_x_list.append(cq)
                    ans_filepath = os.path.join(data_dir, filename[:-3] + 'a')
                    tmp_a = pickle.load(open(ans_filepath, 'rb'))
                    train_a_list.append(tmp_a)
                start+=1
                print(start)
            train_x_vector = np.array(train_x_list)
            train_y_vector = np.array(train_a_list)
            train_x_vector = train_x_vector.reshape(len(train_ctx_len),
                                                    max_ctx_len + max_q_len, columnsize)
            train_y_vector = train_y_vector.reshape(len(train_a_list), columnsize)
            print('\n\n' + 'Length of each Training Context is....')
            print(train_ctx_len)
            print('Number of contexts read is ', len(train_ctx_len))
            print('\n\n' + 'Length of each Training Query is....')
            print(train_q_len)
            print('Number of queries read is ', len(train_q_len))
            print('\n\n')

            print('\n\n'+'Train data X-vector shape is %s' % str(train_x_vector.shape))
            print('\n\n'+'Train data Y-vector shape is %s' % str(train_y_vector.shape))

            pickle.dump(train_x_vector, open('train_x_vector.pickle'+str(count), 'a+b'))
            pickle.dump(train_y_vector, open('train_y_vector.pickle'+str(count), 'a+b'))
            count+=1
            print('\n\n' + 'Pickle number ', count, ' is stored')

            print('\n\n'+'Number of train pickles stored are ', count)

        count = 0
        n_test_contexts = 300
        start = 0
        for i in range(len(test_ctxfiles) // n_test_contexts):
            end = start + n_test_contexts  # number of contexts to be stored
            test_ctx_len = []
            test_q_len = []
            test_a_list = []
            test_x_list = []

            while (start < end):
                filename = test_ctxfiles[start]
                ctx_filepath = os.path.join(data_dir, filename)
                tmp_ct = pickle.load(open(ctx_filepath, 'rb'))
                query_filepath = os.path.join(data_dir, filename[:-3] + 'q')
                tmp_qt = pickle.load(open(query_filepath, 'rb'))
                if (len(tmp_ct) <= max_ctx_len) and \
                        (len(tmp_qt) <= max_q_len) and \
                        (len(tmp_ct) >= min_ctx_len) and \
                        (len(tmp_qt) >= min_q_len):
                    test_ctx_len.append(len(tmp_ct))
                    test_q_len.append(len(pickle.load(open(query_filepath, 'rb'))))
                    ct = np.array(tmp_ct)
                    qt = np.array(tmp_qt)
                    qct = np.concatenate((ct, qt), axis=0)
                    if ((max_q_len+max_ctx_len) - (len(tmp_qt)+len(tmp_ct)) > 0):
                        cqtpad = np.zeros(((max_q_len+max_ctx_len) -
                                           (len(tmp_qt)+len(tmp_ct)), columnsize))
                        qct = np.concatenate((qct, cqtpad), axis=0)
                    test_x_list.append(qct)
                    ans_filepath = os.path.join(data_dir, filename[:-3] + 'a')
                    tmp_at = pickle.load(open(ans_filepath, 'rb'))
                    test_a_list.append(tmp_at)
                start += 1
                print(start)

            test_x_vector = np.array(test_x_list)
            test_y_vector = np.array(test_a_list)
            test_x_vector = test_x_vector.reshape(len(test_ctx_len),
                                                  max_ctx_len + max_q_len, columnsize)
            test_y_vector = test_y_vector.reshape(len(test_a_list), columnsize)
            print('Test data X-shape is %s' % str(test_x_vector.shape))
            print('Test data Y-shape is %s' % str(test_y_vector.shape))
            print('\n\n' + 'Length of each Testing Context is....')
            print(test_ctx_len)
            print('Number of contexts read is ', len(test_ctx_len))
            print('\n\n' + 'Length of each Testing Query is....')
            print(test_q_len)
            print('Number of queries read is ', len(test_q_len))
            print('\n\n')

            pickle.dump(test_x_vector, open('test_x_vector.pickle'+str(count), 'a+b'))
            pickle.dump(test_y_vector, open('test_y_vector.pickle'+str(count), 'a+b'))
            count+=1
            print('\n\n'+'Number of test pickles stored are ', count)

#    return train_x_vector, train_y_vector, test_x_vector, test_y_vector


def create_inputs_for_prediction(pred_dir):
    test_pctx_len = []
    test_pq_len = []
    test_px_list = []
    columnsize = 0
    test_px_vector = []

    test_pctxfiles = [name for name in os.listdir(pred_dir) if name.endswith('_ctx') and name.startswith('test_')]
    test_pqueryfiles = [name for name in os.listdir(pred_dir) if name.endswith('_q') and name.startswith('test_')]

    print('finding the max and min length of context and query files....')
    # find the maxlength of number of words in context and query files. It will be used to create the input shape.
    for filename in test_pctxfiles:
        filepath = os.path.join(pred_dir, filename)
        tmp = pickle.load(open(filepath, 'rb'))
        test_pctx_len.append(len(tmp))
        columnsize = len(tmp[0])

    for filename in test_pqueryfiles:
        filepath = os.path.join(pred_dir, filename)
        test_pq_len.append(len(pickle.load(open(filepath, 'rb'))))


#    max_pctx_len = max(test_pctx_len)
    max_pctx_len = 250
    max_pq_len = max(test_pq_len)

    print('max train context length is %d and min train context length is %d' % (max_pctx_len, min(test_pctx_len)))
    print('max train query length is %d and min train query length is %d' % (max_pq_len, min(test_pq_len)))


    test_pctxfiles = [name for name in os.listdir(pred_dir)
                      if name.endswith('_ctx') and name.startswith('test_')]
    test_pqueryfiles = [name for name in os.listdir(pred_dir)
                        if name.endswith('_q') and name.startswith('test_')]

    max_pctx_len = max(test_pctx_len)
    max_pq_len = max(test_pq_len)
    min_pctx_len = min(test_pctx_len)
    min_pq_len = min(test_pq_len)
    print('Before restriction, max prediction context length is %d and '
          'min train context length is %d' % (max_pctx_len, min_pctx_len))
    print('Before restriction, max prediction query length is %d and '
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
    n_pred_contexts = 50 # max number of context files that could be
    # stored in a single pickle. More contexts will overflow memory
    start = 0
    for i in range(len(test_pctxfiles)//n_pred_contexts):
        end = start+n_pred_contexts  # number of contexts to be stored in a single input pickle
        test_pctx_len = []
        test_pq_len = []
        test_px_list = []
        n_temp = 0
        while (start < end):
            filename = test_pctxfiles[start]
            pctx_filepath = os.path.join(pred_dir, filename)
            tmp_pc = pickle.load(open(pctx_filepath, 'rb'))
            pquery_filepath = os.path.join(pred_dir, filename[:-3] + 'q')
            tmp_pq = pickle.load(open(pquery_filepath, 'rb'))

            if (len(tmp_pc) <= max_pctx_len) and \
                    (len(tmp_pq) <= max_pq_len) and \
                    (len(tmp_pc) >= min_pctx_len) and \
                    (len(tmp_pq) >= min_pq_len):

                columnsize = len(tmp_pc[0])
                test_pctx_len.append(len(tmp_pc))
                test_pq_len.append(len(tmp_pq))
                c = np.array(tmp_pc)
                q = np.array(tmp_pq)
                cq = np.concatenate((c, q), axis=0)
                if ((max_pctx_len+max_pq_len) - (len(tmp_pc)+len(tmp_pq)) > 0):
                    pad_zeros = max_pctx_len + max_pq_len - (len(tmp_pc)+len(tmp_pq))
                    pad = np.zeros((pad_zeros, columnsize))
                    cq = np.concatenate((cq, pad), axis=0)
                test_px_list.append(cq)
            start+=1
            print(start)
        test_px_vector = np.array(test_px_list)
        test_px_vector = test_px_vector.reshape(len(test_pctx_len),
                                                max_pctx_len + max_pq_len, columnsize)
        print('\n\n' + 'Length of each prediction Context is....')
        print(test_pctx_len)
        print('Number of prediction contexts read is ', len(test_pctx_len))
        print('\n\n' + 'Length of each prediction Query is....')
        print(test_pq_len)
        print('Number of prediction queries read is ', len(test_pq_len))
        print('\n\n'+'Prediction data X-vector shape is %s' % str(test_px_vector.shape))

        #os.path.join(pred_dir, 'test_px_vector.pickle'+str(count))
        pickle.dump(test_px_vector, open(os.path.join(pred_dir, 'test_px_vector.pickle'+str(count)), 'a+b'))
        count+=1
        print('\n\n' + 'Pickle number ', count, ' is stored')

        print('\n\n'+'Number of prediction pickles stored are ', count)




def lstm_model0():
    """

    :param X_train: X_train numpy matrix of shape (num_batches, context+query, vocabsize)
    :param y_train: y_train is a numpy matrix of shape
    :param X_test:
    :param y_test:
    :return:
    """
    n_epochs = 2
    n_batch_size = 128
    LSTM_size = 4
    learining_rate = 0.001
    decay_rate = 0.5
    activation_func = 'softmax'
    loss_func = 'categorical_crossentropy'
    val_split = 0.1

    print('\n\n' + 'Training LSTM 0...')

    train_x_pickles = [name for name in os.listdir(pickle_dir) if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(pickle_dir) if name.startswith('train_y_')]
    print('Number of LSTM0 train input pickles read are ', len(train_x_pickles))
    print('Number of LSTM0 train label pickles read are ', len(train_y_pickles))


# (os.path.join(model_dir, 'Mahabharat_LSTM0_model_working_V10_AXX.h5')


    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(os.path.join(pickle_dir, X_filename), 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(os.path.join(pickle_dir, y_filename), 'rb'))
        print('\n\n'+'Pickle number ', i, ' loaded for training')
        if os.path.exists(os.path.join(model_dir, 'Mahabharat_LSTM0_model_working_V10_AXX.h5')):
            model = load_model(os.path.join(model_dir, 'Mahabharat_LSTM0_model_working_V10_AXX.h5'))
            print('Model loaded...')
        else:
            model = Sequential()
            model.add(LSTM(LSTM_size, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dense(X_train.shape[2], kernel_regularizer=regularizers.l2(0.01),
                            activation='tanh'))
            model.add(Activation(activation_func))

            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=decay_rate)
            model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy', 'mae'])
            print('model compiled')

        model.fit(X_train, y_train, epochs=n_epochs,
                     batch_size=n_batch_size,validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
#        weights = model.get_weights()

#        print(model.history)
        model.save(os.path.join(model_dir, 'Mahabharat_LSTM0_model_working_V10_AXX.h5'))
        del model

    test_x_pickles = [name for name in os.listdir(pickle_dir)
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(pickle_dir)
                      if name.startswith('test_y_')]
    print('\n\n'+'Number of LSTM0 test input pickles read are ', len(test_x_pickles))
    print('\n\n'+'Number of LSTM0 test label pickles read are ', len(test_y_pickles))

    print('\n\n'+'Testing started...')

# os.path.join(model_dir, X_filename)


    if not os.path.exists(os.path.join(model_dir, 'Mahabharat_LSTM0_model_working_V10_AXX.h5')):
        print("Saved model is missing...check")

    else:
        model = load_model(os.path.join(model_dir, 'Mahabharat_LSTM0_model_working_V10_AXX.h5'))
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(os.path.join(pickle_dir, X_filename), 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(os.path.join(pickle_dir, y_filename), 'rb'))
            print('\n\n'+'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)

        score, acc, mae = model.evaluate(X_test, y_test, batch_size=n_batch_size)

    test_px_pickles = [name for name in os.listdir(pickle_dir) if name.startswith('train_px_')]
    print('Number of LSTM0 prediction input pickles read are ', len(test_px_pickles))

    X_pred = []

    model = load_model(os.path.join(model_dir, 'Mahabharat_LSTM0_model_working_V10_AXX.h5'))
    for i in range(len(test_px_pickles)):
        X_filename = test_px_pickles[i]
        X_pred = pickle.load(open(os.path.join(pickle_dir, X_filename), 'rb'))
        model.predict(X_pred, batch_size=32, verbose=1)
    pred = model.predict(X_pred, batch_size=32, verbose=1)
    print('Prediction values are ', pred)

    print('Optimizer: RMSProp')
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', LSTM_size)
    print('Learning rate: ', learining_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Validation split: ', val_split)
    print('Decay rate: ', decay_rate)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')




def lstm_model1():
    """
    creates a unidirectional model
    :param dirpath:
    :return:
    """

    n_epochs = 3
    n_batch_size = 64
    LSTM_size = 32
    learining_rate = 0.001
    decay_rate = 0.2
    activation_func = 'softmax'
    loss_func = 'categorical_crossentropy'
    val_split = 0.1

    print('\n\n' + 'Training LSTM 1...')


    train_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_y_')]
    print('Number of LSTM1 train input pickles read are ', len(train_x_pickles))
    print('Number of LSTM1 train label pickles read are ', len(train_y_pickles))

    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(X_filename, 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(y_filename, 'rb'))
        print('\n\n' + 'Pickle number ', i, ' loaded for training')
        if os.path.exists('Mahabharat_LSTM1_model_working_V10_AXX.h5'):
            model = load_model('Mahabharat_LSTM1_model_working_V10_AXX.h5')
            print('Model loaded...')
        else:

            model = Sequential()
            model.add(LSTM(LSTM_size, input_shape=(X_train.shape[1], X_train.shape[2]),
                           activation='tanh', recurrent_activation='hard_sigmoid',
                           use_bias=True, kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal', bias_initializer='zeros',
                           unit_forget_bias=True, kernel_regularizer=None,
                           recurrent_regularizer=None, bias_regularizer=None,
                           activity_regularizer=None, kernel_constraint=None,
                           recurrent_constraint=None, bias_constraint=None,
                           dropout=0.3, recurrent_dropout=0.2))
            model.add(Dense(X_train.shape[2]))
            model.add(Activation(activation_func))

            optimizer = RMSprop(lr=learining_rate, rho=0.9, epsilon=1e-08, decay=decay_rate)
            model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy','mae'])
            print('model compiled')


        model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
        model.save('Mahabharat_LSTM1_model_working_V10_AXX.h5')
        del model

    test_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_y_')]
    print('\n\n'+'Number of LSTM1 test input pickles read are ', len(test_x_pickles))
    print('\n\n'+'Number of LSTM1 test label pickles read are ', len(test_y_pickles))

    print('\n\n'+'Testing started...')

    if not os.path.exists('Mahabharat_LSTM1_model_working_V10_AXX.h5'):
        print("Saved model is missing...check")

    else:
        model = load_model('Mahabharat_LSTM1_model_working_V10_AXX.h5')
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(X_filename, 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(y_filename, 'rb'))
            print('\n\n'+'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)
        score, acc, mae = model.evaluate(X_test, y_test, batch_size=n_batch_size)





    print('Optimizer: RMSProp')
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', LSTM_size)
    print('Learning rate: ', learining_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Validation split: ', val_split)
    print('Decay rate: ', decay_rate)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')



def lstm_model2():
    """
    creates a unidirectional model
    :param dirpath:
    :return:
    """

    n_epochs = 3
    n_batch_size = 32
    LSTM_size = 64
    learining_rate = 0.001
    decay_rate = 0.2
    activation_func = 'softmax'
    loss_func = 'categorical_crossentropy'
    val_split = 0.1

    print('\n\n' + 'Training LSTM 2...')


    train_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_y_')]
    print('Number of LSTM2 train input pickles read are ', len(train_x_pickles))
    print('Number of LSTM2 train label pickles read are ', len(train_y_pickles))

    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(X_filename, 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(y_filename, 'rb'))
        print('\n\n'+'Pickle number ', i, ' loaded for training')
        if os.path.exists('Mahabharat_LSTM2_model_working_V10_AXX.h5'):
            model = load_model('Mahabharat_LSTM2_model_working_V10_AXX.h5')
            print('Model loaded...')
        else:

            model = Sequential()
            model.add(LSTM(LSTM_size, input_shape=(X_train.shape[1], X_train.shape[2]),
                           activation='softmax', recurrent_activation='relu',
                           use_bias=True, kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal', bias_initializer='RandomUniform',
                           unit_forget_bias=True, kernel_regularizer=regularizers.l2(0.01),
                           recurrent_regularizer=None, bias_regularizer=None,
                           activity_regularizer=None, kernel_constraint=None,
                           recurrent_constraint=None, bias_constraint=None,
                           dropout=0.4, recurrent_dropout=0.4))
            model.add(Dense(X_train.shape[2]))
            model.add(Activation(activation_func))

            optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)
            model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy','mae'])
            print('model compiled')


        model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size,
                             validation_split=val_split,
                             verbose=1, callbacks=None, validation_data=None, shuffle=True,
                             class_weight=None, sample_weight=None, initial_epoch=0)

        model.save('Mahabharat_LSTM2_model_working_V10_AXX.h5')
        del model

    test_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_y_')]
    print('\n\n' + 'Number of LSTM2 test input pickles read are ', len(test_x_pickles))
    print('\n\n' + 'Number of LSTM2 test label pickles read are ', len(test_y_pickles))

    print('\n\n' + 'Testing started...')

    if not os.path.exists('Mahabharat_LSTM2_model_working_V10_AXX.h5'):
        print("Saved model is missing...check")

    else:
        model = load_model('Mahabharat_LSTM2_model_working_V10_AXX.h5')
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(X_filename, 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(y_filename, 'rb'))
            print('\n\n' + 'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)

        score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)

    print('Optimizer: ', optimizer)
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', LSTM_size)
    print('Learning rate: ', learining_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Validation split: ', val_split)
    print('Decay rate: ', decay_rate)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')

def lstm_model3():
    """
    creates a unidirectional model
    :param dirpath:
    :return:
    """

    n_epochs = 3
    n_batch_size = 64
    LSTM_size = 32
    learining_rate = 0.0001
    decay_rate = 0.2
    activation_func = 'softmax'
    loss_func = 'categorical_crossentropy'
    val_split = 0.1

    print('\n\n' + 'Training LSTM 3...')
    train_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_y_')]
    print('Number of LSTM3 train input pickles read are ', len(train_x_pickles))
    print('Number of LSTM3 train label pickles read are ', len(train_y_pickles))


    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(X_filename, 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(y_filename, 'rb'))
        print('\n\n'+'Pickle number ', i, ' loaded for training')
        if os.path.exists('Mahabharat_LSTM3_model_working_V10_AXX.h5'):
            model = load_model('Mahabharat_LSTM3_model_working_V10_AXX.h5')
            print('Model loaded...')
        else:
            model = Sequential()
            model.add(LSTM(LSTM_size, input_shape=(X_train.shape[1], X_train.shape[2]),
                           recurrent_activation='tanh',
                           use_bias=True, kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal', bias_initializer='Ones',
                           unit_forget_bias=True, kernel_regularizer=regularizers.l2(0.02),
                           recurrent_regularizer=None, bias_regularizer=regularizers.l1(0.01),
                           activity_regularizer=None, kernel_constraint=None,
                           recurrent_constraint=None, bias_constraint=None,
                           dropout=0.5, recurrent_dropout=0.0))
            model.add(Dense(X_train.shape[2]))
            model.add(Activation(activation_func))

            optimizer = Adagrad(lr=0.001, epsilon=1e-08, decay=decay_rate)
            model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy','mae'])
            print('model compiled')


        model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)

        model.save('Mahabharat_LSTM3_model_working_V10_AXX.h5')
        del model

    test_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_y_')]
    print('\n\n'+'Number of LSTM3 test input pickles read are ', len(test_x_pickles))
    print('\n\n'+'Number of LSTM3 test label pickles read are ', len(test_y_pickles))

    print('\n\n'+'Testing started...')

    if not os.path.exists('Mahabharat_LSTM3_model_working_V10_AXX.h5'):
        print("Saved model is missing...check")

    else:
        model = load_model('Mahabharat_LSTM3_model_working_V10_AXX.h5')
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(X_filename, 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(y_filename, 'rb'))
            print('\n\n'+'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)

        score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)

    print('Optimizer: ', optimizer)
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', LSTM_size)
    print('Learning rate: ', learining_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Decay rate: ', decay_rate)
    print('Validation split: ', val_split)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')


def bi_lstm_model0():
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    n_epochs = 3
    n_batch_size = 32
    BiLSTM_size = 32
    dropout_rate = 0.1
    decay_rate = 0.3
    activation_func = 'softmax'
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)
    loss_func = 'categorical_crossentropy'
    val_split = 0.1

    print('\n\n' + 'Training BiLSTM0...')

    train_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_y_')]
    print('Number of BiLSTM0 train input pickles read are ', len(train_x_pickles))
    print('Number of BiLSTM0 train label pickles read are ', len(train_y_pickles))

    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(X_filename, 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(y_filename, 'rb'))
        print('\n\n'+'Pickle number ', i, ' loaded for training')
        if os.path.exists('Mahabharat_BiLSTM0_model_working_V10_AXX.h5'):
            model = load_model('Mahabharat_BiLSTM0_model_working_V10_AXX.h5')
            print('Model loaded...')
        else:

            model = Sequential()
            model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1],
                                                                    X_train.shape[2])))
            model.add(Dropout(dropout_rate))
            model.add(Dense(X_train.shape[2]))
            model.add(Activation(activation_func))

            # try using different optimizers and different optimizer configs
            model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


        model.fit(X_train, y_train, batch_size=n_batch_size, epochs=n_epochs,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
#        print(hist.history)
        model.save('Mahabharat_BiLSTM0_model_working_V10_AXX.h5')
        del model

    test_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_y_')]
    print('\n\n'+'Number of BiLSTM0 test input pickles read are ', len(test_x_pickles))
    print('\n\n'+'Number of BiLSTM0 test label pickles read are ', len(test_y_pickles))

    print('\n\n'+'Testing started...')

    if not os.path.exists('Mahabharat_BiLSTM0_model_working_V10_AXX.h5'):
        print("Saved model is missing...check")

    else:
        model = load_model('Mahabharat_BiLSTM0_model_working_V10_AXX.h5')
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(X_filename, 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(y_filename, 'rb'))
            print('\n\n'+'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)
        score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)


    print('Optimizer: ', opt)
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', BiLSTM_size)
    print('Dropout rate: ', dropout_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Decay rate: ', decay_rate)
    print('Validation split: ', val_split)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')



def bi_lstm_model1():
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    n_epochs = 3
    n_batch_size = 32
    BiLSTM_size = 64
    dropout_rate = 0.2
    decay_rate = 0.2
    activation_func = 'softmax'
    opt = Adagrad(lr=0.0001, epsilon=1e-08, decay=decay_rate)
    loss_func = 'categorical_crossentropy'
    val_split = 0.1

    print('\n\n' + 'Training BiLSTM 1...')
    train_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_y_')]
    print('Number of BiLSTM1 train input pickles read are ', len(train_x_pickles))
    print('Number of BiLSTM1 train label pickles read are ', len(train_y_pickles))


    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(X_filename, 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(y_filename, 'rb'))
        print('\n\n'+'Pickle number ', i, ' loaded for training')
        if os.path.exists('Mahabharat_BiLSTM1_model_working_V10_AXX.h5'):
            model = load_model('Mahabharat_BiLSTM1_model_working_V10_AXX.h5')
            print('Model loaded...')
        else:
            model = Sequential()
            model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1],
                                                                    X_train.shape[2])))
            model.add(Dropout(dropout_rate))
            model.add(Dense(X_train.shape[2]))
            model.add(Activation(activation_func))

            # try using different optimizers and different optimizer configs
            model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


        model.fit(X_train, y_train,batch_size=n_batch_size,epochs=n_epochs,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None,
                     shuffle=True,
                     class_weight=None, sample_weight=None,
                     initial_epoch=0)
        model.save('Mahabharat_BiLSTM1_model_working_V10_AXX.h5')
        del model

#    print('model fit successfully')
    test_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_y_')]
    print('\n\n' + 'Number of LSTM0 test input pickles read are ', len(test_x_pickles))
    print('\n\n' + 'Number of LSTM0 test label pickles read are ', len(test_y_pickles))

    print('\n\n' + 'Testing started...')

    if not os.path.exists('Mahabharat_BiLSTM1_model_working_V10_AXX.h5'):
        print("Saved model is missing...check")

    else:
        model = load_model('Mahabharat_BiLSTM1_model_working_V10_AXX.h5')
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(X_filename, 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(y_filename, 'rb'))
            print('\n\n' + 'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)

        score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)



    print('Optimizer: ', opt)
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', BiLSTM_size)
    print('Dropout rate: ', dropout_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Decay rate: ', decay_rate)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')



def bi_lstm_model2():
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    n_epochs = 3
    n_batch_size = 32
    BiLSTM_size = 128
    dropout_rate = 0.3
    decay_rate = 0.2
    activation_func = 'softmax'
    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=decay_rate)
    loss_func = 'kullback_leibler_divergence'
    val_split = 0.1

    print('\n\n' + 'Training BiLSTM 2...')
    train_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_y_')]
    print('Number of BiLSTM2 train input pickles read are ', len(train_x_pickles))
    print('Number of BiLSTM2 train label pickles read are ', len(train_y_pickles))


    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(X_filename, 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(y_filename, 'rb'))
        print('\n\n'+'Pickle number ', i, ' loaded for training')
        if os.path.exists('Mahabharat_BiLSTM2_model_working_V10_AXX.h5'):
            model = load_model('Mahabharat_BiLSTM2_model_working_V10_AXX.h5')
            print('Model loaded...')
        else:
            model = Sequential()
            model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(dropout_rate))
            model.add(Dense(X_train.shape[2]))
            model.add(Activation(activation_func))

            # try using different optimizers and different optimizer configs
            model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


        model.fit(X_train, y_train,batch_size=n_batch_size,epochs=n_epochs,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None,
                     shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)

        model.save('Mahabharat_BiLSTM2_model_working_V10_AXX.h5')
        del model

    test_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_y_')]
    print('\n\n'+'Number of BiLSTM2 test input pickles read are ', len(test_x_pickles))
    print('\n\n'+'Number of BiLSTM2 test label pickles read are ', len(test_y_pickles))

    print('\n\n'+'Testing started...')

    if not os.path.exists('Mahabharat_BiLSTM2_model_working_V10_AXX.h5'):
        print("Saved model is missing...check")

    else:
        model = load_model('Mahabharat_BiLSTM2_model_working_V10_AXX.h5')
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(X_filename, 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(y_filename, 'rb'))
            print('\n\n'+'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)

        score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)


    print('Optimizer: ', opt)
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', BiLSTM_size)
    print('Dropout rate: ', dropout_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Decay rate: ', decay_rate)
    print('Validation split: ', val_split)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')




def bi_lstm_model3():
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    n_epochs = 3
    n_batch_size = 32
    BiLSTM_size = 64
    dropout_rate = 0.4
    decay_rate = 0.3
    activation_func = 'softmax'
    opt = Adagrad(lr=0.001, epsilon=1e-08, decay=decay_rate)
    loss_func = 'kullback_leibler_divergence'
    val_split = 0.1

    print('\n\n' + 'Training BiLSTM 3...')
    train_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_y_')]
    print('Number of BiLSTM3 train input pickles read are ', len(train_x_pickles))
    print('Number of BiLSTM3 train label pickles read are ', len(train_y_pickles))


    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(X_filename, 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(y_filename, 'rb'))
        print('\n\n'+'Pickle number ', i, ' loaded for training')
        if os.path.exists('Mahabharat_BiLSTM3_model_working_V10_AXX.h5'):
            model = load_model('Mahabharat_BiLSTM3_model_working_V10_AXX.h5')
            print('Model loaded...')
        else:
            model = Sequential()
            model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1],
                                                                    X_train.shape[2])))
            model.add(Dropout(dropout_rate))
            model.add(Dense(X_train.shape[2]))
            model.add(Activation(activation_func))

            # try using different optimizers and different optimizer configs
            model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


        model.fit(X_train, y_train,batch_size=n_batch_size,
                     epochs=n_epochs,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None,
                     shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)

        model.save('Mahabharat_BiLSTM3_model_working_V10_AXX.h5')
        del model

    test_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_y_')]
    print('\n\n' + 'Number of BiLSTM3 test input pickles read are ', len(test_x_pickles))
    print('\n\n' + 'Number of BiLSTM3 test label pickles read are ', len(test_y_pickles))

    print('\n\n' + 'Testing started...')

    if not os.path.exists('Mahabharat_BiLSTM3_model_working_V10_AXX.h5'):
        print("Saved model is missing...check")

    else:
        model = load_model('Mahabharat_BiLSTM3_model_working_V10_AXX.h5')
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(X_filename, 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(y_filename, 'rb'))
            print('\n\n' + 'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)

        score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)


    print('Optimizer: ', opt)
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', BiLSTM_size)
    print('Dropout rate: ', dropout_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Decay rate: ', decay_rate)
    print('Validation split: ', val_split)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')



def bi_lstm_model4():
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    n_epochs = 3
    n_batch_size = 32
    BiLSTM_size = 16
    dropout_rate = 0.1
    val_split = 0.1
    decay_rate = 0.3
    activation_func = 'softmax'
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)
    loss_func = 'categorical_crossentropy'

    print('\n\n' + 'Training BiLSTM 4...')
    train_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_x_')]
    train_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('train_y_')]
    print('Number of BiLSTM4 train input pickles read are ', len(train_x_pickles))
    print('Number of BiLSTM4 train label pickles read are ', len(train_y_pickles))


    for i in range(len(train_x_pickles)):
        X_filename = train_x_pickles[i]
        X_train = pickle.load(open(X_filename, 'rb'))
        y_filename = train_y_pickles[i]
        y_train = pickle.load(open(y_filename, 'rb'))
        print('\n\n'+'Pickle number ', i, ' loaded for training')
        if os.path.exists('Mahabharat_BiLSTM4_model_working_V10_AXX.h5'):
            model = load_model('Mahabharat_BiLSTM4_model_working_V10_AXX.h5')
            print('Model loaded...')
        else:
            model = Sequential()
            model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1],
                                                                    X_train.shape[2])))
            model.add(Dropout(dropout_rate))
            model.add(Dense(X_train.shape[2]))
            model.add(Activation(activation_func))

            # try using different optimizers and different optimizer configs
            model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


        model.fit(X_train, y_train,batch_size=n_batch_size,
                     epochs=n_epochs,validation_split=val_split,
                     verbose=1, callbacks=None,
                     validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None,
                     initial_epoch=0)

        model.save('Mahabharat_BiLSTM4_model_working_V10_AXX.h5')
        del model

    test_x_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_x_')]
    test_y_pickles = [name for name in os.listdir(os.path.abspath(''))
                      if name.startswith('test_y_')]
    print('\n\n' + 'Number of BiLSTM4 test input pickles read are ', len(test_x_pickles))
    print('\n\n' + 'Number of BiLSTM4 test label pickles read are ', len(test_y_pickles))

    print('\n\n' + 'Testing started...')

    if not os.path.exists('Mahabharat_BiLSTM4_model_working_V10_AXX.h5'):
        print("Saved model is missing...check")

    else:
        model = load_model('Mahabharat_BiLSTM4_model_working_V10_AXX.h5')
        for i in range(len(test_x_pickles)):
            X_filename = test_x_pickles[i]
            X_test = pickle.load(open(X_filename, 'rb'))
            y_filename = test_y_pickles[i]
            y_test = pickle.load(open(y_filename, 'rb'))
            print('\n\n' + 'Pickle number ', i, ' loaded for testing')
            model.evaluate(X_test, y_test, batch_size=n_batch_size)

        score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)


    print('Optimizer: ', opt)
    print('epochs: ', n_epochs)
    print('Batch_size: ', n_batch_size)
    print('LSTM Size: ', BiLSTM_size)
    print('Dropout rate: ', dropout_rate)
    print('Loss computation: ', loss_func)
    print('Activation: ', activation_func)
    print('Decay rate: ', decay_rate)
    print('Validation split: ', val_split)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('\n\n')





if __name__ == '__main__':
    my_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    pred_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'predict'))
    pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pickle'))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))

    create_inputs(data_dir)
    create_inputs_for_prediction(pred_dir)

    '''
    print('X_train_shape is')
    print(X_train.shape)
    print('y_train_shape is')
    print(y_train.shape)
    print('X_test_shape is')
    print(X_test.shape)
    print('y_test_shape is')
    print(y_test.shape)
    '''

    lstm_model0()

    '''
    lstm_model1()
    lstm_model2()
    lstm_model3()
    bi_lstm_model0()
    bi_lstm_model1()
    bi_lstm_model2()
    bi_lstm_model3()
    bi_lstm_model4()
    '''


'''

#Here is teh directory path; create a subdiretory called 'predict' at the same level of 'data'.
# in other words, create a subdirectory at parent directory

#     dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'predict'))



# Create inputs for prediction



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



'''