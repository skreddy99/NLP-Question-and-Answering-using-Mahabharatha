import keras
from keras.layers import LSTM, Dense, Activation, Bidirectional, Dropout
from keras.optimizers import RMSprop, Adam, Adagrad, SGD
import datautils as dt
from keras.models import Sequential
from keras import regularizers
import os
import numpy as np
import pickle
from keras.models import load_model

def create_inputs(dirpath):
    """

    :param dirpath:
    :return:
    """
    train_a_list = []
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
#    n_bytes = 2 ** 31
#    max_bytes = 2 ** 31 - 1


    train_ctxfiles = [name for name in os.listdir(dirpath)
                      if name.endswith('_ctx') and name.startswith('train_')]
#    train_queryfiles = [name for name in os.listdir(dirpath)
#                        if name.endswith('_q') and name.startswith('train_')]
    test_ctxfiles = [name for name in os.listdir(dirpath)
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

    max_ctx_len = 280
    max_q_len = 20
    min_ctx_len = 200
    min_q_len = 5
    print('Max ctx len is ', max_ctx_len)
    print('\n\n')
    print('Max q len is ', max_q_len)
    print('\n\n')
    print('Min ctx len is ', min_ctx_len)
    print('\n\n')
    print('Min q len is ', min_q_len)
    print('\n\n')

    file_exists = 0
    if os.path.exists('train_x_vector.pickle'):
        file_exists +=1
        train_x_vector = np.array(pickle.load(open('train_x_vector.pickle', 'rb')))

    if os.path.exists('train_y_vector.pickle'):
        file_exists +=1
        train_y_vector = pickle.load(open('train_y_vector.pickle', 'rb'))

    if os.path.exists('test_x_vector.pickle'):
        file_exists +=1
        test_x_vector = pickle.load(open('test_x_vector.pickle', 'rb'))


    if os.path.exists('test_y_vector.pickle'):
        file_exists +=1
        test_y_vector = pickle.load(open('test_y_vector.pickle', 'rb'))

    if file_exists < 4:
        for filename in train_ctxfiles:
            ctx_filepath = os.path.join(dirpath, filename)
            tmp_c = pickle.load(open(ctx_filepath, 'rb'))
            query_filepath = os.path.join(dirpath, filename[:-3] + 'q')
            tmp_q = pickle.load(open(query_filepath, 'rb'))

            if (len(tmp_c) <= max_ctx_len) and \
                    (len(tmp_q) <= max_q_len) and \
                    (len(tmp_c) >= min_ctx_len) and \
                    (len(tmp_q) >= min_q_len):

                columnsize = len(tmp_c[0])
                train_ctx_len.append(len(tmp_c))
                train_q_len.append(len(pickle.load(open(query_filepath, 'rb'))))

                c = np.array(tmp_c)
                q = np.array(tmp_q)

                cq = np.concatenate((c, q), axis=0)

                if ((max_ctx_len+max_q_len) - (len(tmp_c)+len(tmp_q)) > 0):
                    pad_zeros = max_ctx_len + max_q_len - (len(tmp_c)+len(tmp_q))
                    pad = np.zeros((pad_zeros, columnsize))
                    cq = np.concatenate((cq, pad), axis=0)

                train_x_list.append(cq)

                ans_filepath = os.path.join(dirpath, filename[:-3] + 'a')
                tmp_a = pickle.load(open(ans_filepath, 'rb'))
                train_a_list.append(tmp_a)

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

        print('Train data X-vector shape is %s' % str(train_x_vector.shape))
        print('Train data Y-vector shape is %s' % str(train_y_vector.shape))

#        pickle.dump(train_x_vector, open('train_x_vector.pickle', 'a+b'), pickle.HIGHEST_PROTOCOL)
#        pickle.dump(train_y_vector, open('train_y_vector.pickle', 'a+b'), pickle.HIGHEST_PROTOCOL)


#from here

        for filename in test_ctxfiles:
            ctx_filepath = os.path.join(dirpath, filename)
            tmp_ct = pickle.load(open(ctx_filepath, 'rb'))
            query_filepath = os.path.join(dirpath, filename[:-3] + 'q')
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

                ans_filepath = os.path.join(dirpath, filename[:-3] + 'a')
                tmp_at = pickle.load(open(ans_filepath, 'rb'))
                test_a_list.append(tmp_at)

# Till here

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


#        pickle.dump(test_x_vector, open('test_x_vector.pickle', 'a+b'), pickle.HIGHEST_PROTOCOL)
#        pickle.dump(test_y_vector, open('test_y_vector.pickle', 'a+b'), pickle.HIGHEST_PROTOCOL)

    return train_x_vector, train_y_vector, test_x_vector, test_y_vector


def lstm_model0(X_train, y_train, X_test, y_test):
    """

    :param X_train: X_train numpy matrix of shape (num_batches, context+query, vocabsize)
    :param y_train: y_train is a numpy matrix of shape
    :param X_test:
    :param y_test:
    :return:
    """
    n_epochs = 5
    n_batch_size = 16
    LSTM_size = 4
    learining_rate = 0.001
    decay_rate = 0.5
    activation_func = 'softmax'
    loss_func = 'categorical_crossentropy'
    val_split = 0.2


    print('\n\n' + 'Training LSTM 0...')
    model = Sequential()
    model.add(LSTM(LSTM_size, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(X_train.shape[2],kernel_regularizer=regularizers.l2(0.01),
                    activation='tanh'))
    model.add(Activation(activation_func))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=decay_rate)
    model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy','mae'])
    print('model compiled')

    hist = model.fit(X_train, y_train, epochs=n_epochs,
                     batch_size=n_batch_size,validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
    print(hist.history)

    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test, batch_size=n_batch_size)

    model.save('Mahabharat_LSTM0_working_V9_A59.h5')
    del model

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




def lstm_model1(X_train, y_train, X_test, y_test):
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
    val_split = 0.2

    print('\n\n' + 'Training LSTM 1...')
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


    hist = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
    print(hist.history)

    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)

    model.save('Mahabharat_LSTM1_working_V9_A59.h5')
    del model

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



def lstm_model2(X_train, y_train, X_test, y_test):
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


    hist = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
    print(hist.history)


    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)
    model.save('Mahabharat_LSTM2_working_V9_A59.h5')
    del model

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

def lstm_model3(X_train, y_train, X_test, y_test):
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


    hist = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
    print(hist.history)

    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)
    model.save('Mahabharat_LSTM3_working_V9_A59.h5')
    del model

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


def bi_lstm_model0(X_train, y_train, X_test, y_test):
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

    print('\n\n' + 'Training bidirectional LSTM 0...')
    model = Sequential()
    model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1],
                                                            X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(X_train.shape[2]))
    model.add(Activation(activation_func))

    # try using different optimizers and different optimizer configs
    model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


    hist = model.fit(X_train, y_train, batch_size=n_batch_size, epochs=n_epochs,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
    print(hist.history)

    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)

    model.save('Mahabharat_BiLSTM0_working_V9_A59.h5')
    del model


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



def bi_lstm_model1(X_train, y_train, X_test, y_test):
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

    print('\n\n' + 'Training bidirectional LSTM 1...')
    model = Sequential()
    model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1],
                                                            X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(X_train.shape[2]))
    model.add(Activation(activation_func))

    # try using different optimizers and different optimizer configs
    model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


    hist = model.fit(X_train, y_train,batch_size=n_batch_size,epochs=n_epochs,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None,
                     shuffle=True,
                     class_weight=None, sample_weight=None,
                     initial_epoch=0)
    print(hist.history)

    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)

    model.save('Mahabharat_BiLSTM1_working_V9.h5')
    del model


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



def bi_lstm_model2(X_train, y_train, X_test, y_test):
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

    print('\n\n' + 'Training bidirectional LSTM 2...')
    model = Sequential()
    model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(X_train.shape[2]))
    model.add(Activation(activation_func))

    # try using different optimizers and different optimizer configs
    model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


    hist = model.fit(X_train, y_train,batch_size=n_batch_size,epochs=n_epochs,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None,
                     shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
    print(hist.history)

    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)

    model.save('Mahabharat_BiLSTM2_working_V9_A59.h5')
    del model

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




def bi_lstm_model3(X_train, y_train, X_test, y_test):
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

    print('\n\n' + 'Training bidirectional LSTM 3...')
    model = Sequential()
    model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1],
                                                            X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(X_train.shape[2]))
    model.add(Activation(activation_func))

    # try using different optimizers and different optimizer configs
    model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


    hist = model.fit(X_train, y_train,batch_size=n_batch_size,
                     epochs=n_epochs,
                     validation_split=val_split,
                     verbose=1, callbacks=None, validation_data=None,
                     shuffle=True,
                     class_weight=None, sample_weight=None, initial_epoch=0)
    print(hist.history)

    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)
    model.save('Mahabharat_BiLSTM3_working_V9.h5')
    del model


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



def bi_lstm_model4(X_train, y_train, X_test, y_test):
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

    print('\n\n' + 'Training bidirectional LSTM 4...')
    model = Sequential()
    model.add(Bidirectional(LSTM(BiLSTM_size), input_shape=(X_train.shape[1],
                                                            X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(X_train.shape[2]))
    model.add(Activation(activation_func))

    # try using different optimizers and different optimizer configs
    model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy','mae'])


    hist = model.fit(X_train, y_train,batch_size=n_batch_size,
                     epochs=n_epochs,validation_split=val_split,
                     verbose=1, callbacks=None,
                     validation_data=None, shuffle=True,
                     class_weight=None, sample_weight=None,
                     initial_epoch=0)
    print(hist.history)

    print('model fit successfully')
    score, acc, mae = model.evaluate(X_test, y_test,
                                batch_size=n_batch_size)

    model.save('Mahabharat_BiLSTM4_working_V9_A59.h5')
    del model

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
    dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    X_train, y_train, X_test, y_test = create_inputs(dirpath)
    print('X_train_shape is')
    print(X_train.shape)
    print('y_train_shape is')
    print(y_train.shape)
    print('X_test_shape is')
    print(X_test.shape)
    print('y_test_shape is')
    print(y_test.shape)
    lstm_model0(X_train, y_train, X_test, y_test)
    lstm_model1(X_train, y_train, X_test, y_test)
    lstm_model2(X_train, y_train, X_test, y_test)
    lstm_model3(X_train, y_train, X_test, y_test)
    bi_lstm_model0(X_train, y_train, X_test, y_test)
    bi_lstm_model1(X_train, y_train, X_test, y_test)
    bi_lstm_model2(X_train, y_train, X_test, y_test)
    bi_lstm_model3(X_train, y_train, X_test, y_test)
    bi_lstm_model4(X_train, y_train, X_test, y_test)


