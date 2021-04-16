import tensorflow as tf
from keras.layers import LSTM, Bidirectional, Multiply
from keras.layers.core import *
from keras import backend as K

class LSTM_Attention():
    '''
    用于根据不同的需求构造不同的网络结构
    传入参数：
            LR_RATE //学习率
            LSTM_UNITS  //神经元个数
            IS_Con1D    //是否选用一维卷积
            X_TRAIN     //训练集
            KERNEL_SIZE     //卷积核尺寸
            FILTERS     //卷积核个数
            DROPOUT     //dropout
            CLASS_NUMBER//分类数

    '''
    def __init__(self):
        self.LR_RATE=0.001
        self.DROPOUT=0.4
        self.LSTM_UNITS=64
        self.IS_Con1D=1
        self.CLASS_NUM=2
        #self.X_TRAIN
        self.KERNEL_SIZE=1
        self.FILTERS=19
    def make_model(self,X_TRAIN):
        INPUTS=tf.keras.Input(shape=(len(X_TRAIN[0]),len(X_TRAIN[0][0])))

        if self.IS_Con1D==1:
            Conv_out=tf.keras.layers.Conv1D(filters=self.FILTERS,kernel_size=self.KERNEL_SIZE)(INPUTS)
        else:
            Conv_out=tf.keras.layers.Conv2D(filters=self.FILTERS,kernel_size=self.KERNEL_SIZE)(INPUTS)

        LSTM_out = Bidirectional(LSTM(self.LSTM_UNITS, return_sequences=True))(Conv_out)
        LSTM_out = Dropout(0.3)(LSTM_out)

        # 自定义attention

        time_steps = K.int_shape(LSTM_out)[1]
        # input_dim = K.int_shape(LSTM_out)[2]
        att = Permute((2, 1))(LSTM_out)

        attention=Dense(time_steps,activation='softmax')(att)
        a_probs = Permute((2, 1))(attention)
        # 转置相乘（乘以对应的权重）
        Attention_out=Multiply()([LSTM_out,a_probs])
        Dense_in=Flatten()(Attention_out)
        Dense_out=Dense(self.CLASS_NUM,activation='softmax')(Dense_in)
        lstm_model=tf.keras.Model(inputs=INPUTS,outputs=Dense_out)

        lstm_model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=self.LR_RATE),
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
        return lstm_model


#compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])