import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import *
from sklearn.cross_validation import *
from sklearn.metrics import *

train_file = 'data/small_train.csv'

train_data = np.loadtxt(train_file, skiprows = 1, delimiter = ',', dtype = 'float')
X = train_data[:, 1:]
X = X/255
raw_Y = train_data[:, 0].reshape(-1, 1)

X_train, X_cv, raw_Y_train, raw_Y_cv = train_test_split(X, raw_Y, test_size = 0.20)
Y_expander = OneHotEncoder().fit(raw_Y)
Y_train = Y_expander.transform(raw_Y_train).astype(int).toarray()
Y_cv = Y_expander.transform(raw_Y_cv).astype(int).toarray()

model = Sequential()
model.add(Dense(input_dim = X.shape[1], output_dim = 512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim = 10))
model.add(Activation('softmax'))
# model.add(Activation('relu'))
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', class_mode = 'categorical')

sgd = SGD(lr=0.2, decay=1e-7, momentum=0.1, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd')

model.fit(X_train, Y_train, nb_epoch = 20, batch_size = 20, show_accuracy = True, verbose = 1, validation_split = 0.05)
Y_cv_pred = model.predict_classes(X_cv, batch_size = 20, verbose = 1)

# print('Compare labels and predicted values:')
# print(raw_Y_cv.reshape(-1).astype(int))
# print(Y_cv_pred)

score = accuracy_score(raw_Y_cv, Y_cv_pred)
print('Accuracy score: ', score)
