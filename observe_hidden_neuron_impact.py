"""
This code tries to explore the impact of the number of hidden neurons if the
network architecture is fixed.

From the output, we can see that the benefit of increasing number of hidden
neurons starts to flatten out at 64 hidden neurons and up. Below that, adding
more hidden neurons increases the model performance noticeably.

###############################################################################
Output of this code:

Using Theano backend.
Using gpu device 0: GeForce GTX 780 (CNMeM is enabled)
Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 0.4780 - acc: 0.8723 - val_loss: 0.3780 - val_acc: 0.8893
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 0.3115 - acc: 0.9109 - val_loss: 0.3429 - val_acc: 0.8970
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 0.2814 - acc: 0.9201 - val_loss: 0.3241 - val_acc: 0.9012
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 0.2601 - acc: 0.9271 - val_loss: 0.3074 - val_acc: 0.9119
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 0.2413 - acc: 0.9326 - val_loss: 0.2934 - val_acc: 0.9155
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 0.2236 - acc: 0.9372 - val_loss: 0.2786 - val_acc: 0.9208
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 0.2069 - acc: 0.9426 - val_loss: 0.2570 - val_acc: 0.9220
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 0.1912 - acc: 0.9472 - val_loss: 0.2496 - val_acc: 0.9286
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 0.1771 - acc: 0.9503 - val_loss: 0.2416 - val_acc: 0.9262
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 0.1638 - acc: 0.9549 - val_loss: 0.2237 - val_acc: 0.9393
8400/8400 [==============================] - 0s
Using [512] number of hidden neurons yields. Accuracy score: 0.9435

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 0.4964 - acc: 0.8680 - val_loss: 0.3790 - val_acc: 0.8917
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 0.3119 - acc: 0.9117 - val_loss: 0.3419 - val_acc: 0.8958
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 0.2772 - acc: 0.9208 - val_loss: 0.3153 - val_acc: 0.9065
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 0.2523 - acc: 0.9286 - val_loss: 0.2963 - val_acc: 0.9149
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 0.2296 - acc: 0.9355 - val_loss: 0.2787 - val_acc: 0.9161
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 0.2100 - acc: 0.9410 - val_loss: 0.2590 - val_acc: 0.9220
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 0.1926 - acc: 0.9466 - val_loss: 0.2373 - val_acc: 0.9327
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 0.1769 - acc: 0.9503 - val_loss: 0.2276 - val_acc: 0.9315
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 0.1631 - acc: 0.9544 - val_loss: 0.2203 - val_acc: 0.9333
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 0.1514 - acc: 0.9586 - val_loss: 0.2045 - val_acc: 0.9363
8400/8400 [==============================] - 0s
Using [256] number of hidden neurons yields. Accuracy score: 0.9477

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 0.5073 - acc: 0.8660 - val_loss: 0.3829 - val_acc: 0.8851
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 0.3087 - acc: 0.9128 - val_loss: 0.3273 - val_acc: 0.9018
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 0.2671 - acc: 0.9248 - val_loss: 0.2976 - val_acc: 0.9107
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 0.2378 - acc: 0.9339 - val_loss: 0.2767 - val_acc: 0.9185
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 0.2153 - acc: 0.9397 - val_loss: 0.2580 - val_acc: 0.9238
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 0.1957 - acc: 0.9447 - val_loss: 0.2407 - val_acc: 0.9256
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 0.1795 - acc: 0.9507 - val_loss: 0.2282 - val_acc: 0.9292
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 0.1657 - acc: 0.9538 - val_loss: 0.2127 - val_acc: 0.9357
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 0.1539 - acc: 0.9563 - val_loss: 0.2057 - val_acc: 0.9405
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 0.1436 - acc: 0.9605 - val_loss: 0.1935 - val_acc: 0.9417
8400/8400 [==============================] - 0s
Using [128] number of hidden neurons yields. Accuracy score: 0.9485

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 0.5406 - acc: 0.8565 - val_loss: 0.3921 - val_acc: 0.8869
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 0.3130 - acc: 0.9111 - val_loss: 0.3354 - val_acc: 0.8994
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 0.2674 - acc: 0.9241 - val_loss: 0.3013 - val_acc: 0.9131
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 0.2371 - acc: 0.9320 - val_loss: 0.2794 - val_acc: 0.9196
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 0.2137 - acc: 0.9401 - val_loss: 0.2595 - val_acc: 0.9256
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 0.1951 - acc: 0.9458 - val_loss: 0.2424 - val_acc: 0.9298
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 0.1793 - acc: 0.9506 - val_loss: 0.2301 - val_acc: 0.9339
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 0.1665 - acc: 0.9542 - val_loss: 0.2187 - val_acc: 0.9375
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 0.1552 - acc: 0.9571 - val_loss: 0.2090 - val_acc: 0.9369
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 0.1454 - acc: 0.9598 - val_loss: 0.1977 - val_acc: 0.9411
8400/8400 [==============================] - 0s
Using [64] number of hidden neurons yields. Accuracy score: 0.9483

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 0.5955 - acc: 0.8519 - val_loss: 0.4001 - val_acc: 0.8881
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 0.3237 - acc: 0.9099 - val_loss: 0.3383 - val_acc: 0.9012
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 0.2745 - acc: 0.9230 - val_loss: 0.3088 - val_acc: 0.9107
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 0.2451 - acc: 0.9306 - val_loss: 0.2907 - val_acc: 0.9190
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 0.2240 - acc: 0.9374 - val_loss: 0.2730 - val_acc: 0.9190
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 0.2075 - acc: 0.9421 - val_loss: 0.2655 - val_acc: 0.9214
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 0.1938 - acc: 0.9452 - val_loss: 0.2528 - val_acc: 0.9280
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 0.1823 - acc: 0.9485 - val_loss: 0.2406 - val_acc: 0.9298
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 0.1723 - acc: 0.9517 - val_loss: 0.2383 - val_acc: 0.9357
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 0.1639 - acc: 0.9542 - val_loss: 0.2297 - val_acc: 0.9310
8400/8400 [==============================] - 0s
Using [32] number of hidden neurons yields. Accuracy score: 0.9427

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 0.7089 - acc: 0.8291 - val_loss: 0.4609 - val_acc: 0.8798
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 0.3733 - acc: 0.8984 - val_loss: 0.3816 - val_acc: 0.8887
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 0.3131 - acc: 0.9118 - val_loss: 0.3468 - val_acc: 0.9012
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 0.2805 - acc: 0.9206 - val_loss: 0.3247 - val_acc: 0.9065
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 0.2583 - acc: 0.9258 - val_loss: 0.3058 - val_acc: 0.9077
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 0.2415 - acc: 0.9310 - val_loss: 0.2974 - val_acc: 0.9077
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 0.2281 - acc: 0.9350 - val_loss: 0.2895 - val_acc: 0.9095
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 0.2173 - acc: 0.9383 - val_loss: 0.2761 - val_acc: 0.9190
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 0.2085 - acc: 0.9401 - val_loss: 0.2735 - val_acc: 0.9161
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 0.2002 - acc: 0.9430 - val_loss: 0.2632 - val_acc: 0.9214
8400/8400 [==============================] - 0s
Using [16] number of hidden neurons yields. Accuracy score: 0.9308

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 0.9088 - acc: 0.7632 - val_loss: 0.6142 - val_acc: 0.8369
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 0.4828 - acc: 0.8726 - val_loss: 0.4658 - val_acc: 0.8744
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 0.3959 - acc: 0.8922 - val_loss: 0.4147 - val_acc: 0.8863
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 0.3573 - acc: 0.9011 - val_loss: 0.3937 - val_acc: 0.8881
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 0.3348 - acc: 0.9070 - val_loss: 0.3715 - val_acc: 0.8994
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 0.3185 - acc: 0.9112 - val_loss: 0.3719 - val_acc: 0.9012
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 0.3066 - acc: 0.9133 - val_loss: 0.3681 - val_acc: 0.8935
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 0.2974 - acc: 0.9161 - val_loss: 0.3591 - val_acc: 0.8952
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 0.2901 - acc: 0.9180 - val_loss: 0.3471 - val_acc: 0.8976
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 0.2826 - acc: 0.9204 - val_loss: 0.3526 - val_acc: 0.8994
8400/8400 [==============================] - 0s
Using [8] number of hidden neurons yields. Accuracy score: 0.9089

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 1.3762 - acc: 0.6177 - val_loss: 0.9962 - val_acc: 0.7363
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 0.8565 - acc: 0.7716 - val_loss: 0.7968 - val_acc: 0.7857
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 0.7116 - acc: 0.8124 - val_loss: 0.6857 - val_acc: 0.8161
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 0.6256 - acc: 0.8309 - val_loss: 0.6324 - val_acc: 0.8202
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 0.5807 - acc: 0.8378 - val_loss: 0.5966 - val_acc: 0.8292
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 0.5576 - acc: 0.8400 - val_loss: 0.5833 - val_acc: 0.8310
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 0.5414 - acc: 0.8444 - val_loss: 0.5690 - val_acc: 0.8369
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 0.5312 - acc: 0.8470 - val_loss: 0.5654 - val_acc: 0.8369
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 0.5240 - acc: 0.8492 - val_loss: 0.5586 - val_acc: 0.8363
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 0.5185 - acc: 0.8502 - val_loss: 0.5604 - val_acc: 0.8345
8400/8400 [==============================] - 0s
Using [4] number of hidden neurons yields. Accuracy score: 0.8348

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 1.7218 - acc: 0.3774 - val_loss: 1.4964 - val_acc: 0.3940
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 1.4286 - acc: 0.4109 - val_loss: 1.3737 - val_acc: 0.4357
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 1.3616 - acc: 0.4331 - val_loss: 1.3383 - val_acc: 0.4440
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 1.3256 - acc: 0.4557 - val_loss: 1.3101 - val_acc: 0.4530
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 1.3001 - acc: 0.4776 - val_loss: 1.2965 - val_acc: 0.4970
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 1.2756 - acc: 0.4940 - val_loss: 1.2787 - val_acc: 0.5143
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 1.2548 - acc: 0.5053 - val_loss: 1.2650 - val_acc: 0.5167
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 1.2371 - acc: 0.5121 - val_loss: 1.2501 - val_acc: 0.5238
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 1.2213 - acc: 0.5223 - val_loss: 1.2536 - val_acc: 0.5369
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 1.2034 - acc: 0.5414 - val_loss: 1.2206 - val_acc: 0.5738
8400/8400 [==============================] - 0s
Using [2] number of hidden neurons yields. Accuracy score: 0.5500

Train on 31920 samples, validate on 1680 samples
Epoch 1/10
31920/31920 [==============================] - 2s - loss: 1.9788 - acc: 0.2089 - val_loss: 1.8544 - val_acc: 0.1958
Epoch 2/10
31920/31920 [==============================] - 2s - loss: 1.8294 - acc: 0.2240 - val_loss: 1.8018 - val_acc: 0.2202
Epoch 3/10
31920/31920 [==============================] - 2s - loss: 1.7907 - acc: 0.2356 - val_loss: 1.7807 - val_acc: 0.2274
Epoch 4/10
31920/31920 [==============================] - 2s - loss: 1.7705 - acc: 0.2464 - val_loss: 1.7685 - val_acc: 0.2435
Epoch 5/10
31920/31920 [==============================] - 2s - loss: 1.7586 - acc: 0.2505 - val_loss: 1.7691 - val_acc: 0.2321
Epoch 6/10
31920/31920 [==============================] - 2s - loss: 1.7501 - acc: 0.2488 - val_loss: 1.7620 - val_acc: 0.2518
Epoch 7/10
31920/31920 [==============================] - 2s - loss: 1.7432 - acc: 0.2532 - val_loss: 1.7705 - val_acc: 0.2667
Epoch 8/10
31920/31920 [==============================] - 2s - loss: 1.7363 - acc: 0.2617 - val_loss: 1.7511 - val_acc: 0.2571
Epoch 9/10
31920/31920 [==============================] - 2s - loss: 1.7312 - acc: 0.2637 - val_loss: 1.7445 - val_acc: 0.2738
Epoch 10/10
31920/31920 [==============================] - 2s - loss: 1.7272 - acc: 0.2674 - val_loss: 1.7418 - val_acc: 0.2875
8400/8400 [==============================] - 0s
Using [1] number of hidden neurons yields. Accuracy score: 0.2690

"""

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import *
from sklearn.cross_validation import *
from sklearn.metrics import *

TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'

train_data = np.loadtxt(TRAIN_FILE, skiprows = 1, delimiter = ',', dtype = 'float')
X = train_data[:, 1:]
X = X/255
raw_Y = train_data[:, 0].reshape(-1, 1)

X_test = np.loadtxt(TEST_FILE, skiprows = 1, delimiter = ',', dtype = 'float')
X_test = X_test/255

X_train, X_cv, raw_Y_train, raw_Y_cv = train_test_split(X, raw_Y, test_size = 0.20)
Y_expander = OneHotEncoder().fit(raw_Y)
Y_train = Y_expander.transform(raw_Y_train).astype(int).toarray()
Y_cv = Y_expander.transform(raw_Y_cv).astype(int).toarray()

for n_hidden in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
    model = Sequential()
    model.add(Dense(input_dim = X.shape[1], output_dim = n_hidden))
    model.add(Activation('tanh'))
    model.add(Dense(output_dim = 10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.2, decay=1e-7, momentum=0.1, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    model.fit(X_train, Y_train, nb_epoch = 10, batch_size = 10, show_accuracy = True, verbose = 1, validation_split = 0.05)
    Y_cv_pred = model.predict_classes(X_cv, batch_size = 10, verbose = 1)

    score = accuracy_score(raw_Y_cv, Y_cv_pred)
    print('Using [%d] number of hidden neurons yields. Accuracy score: %.4f' % (n_hidden, score))
    print('')
