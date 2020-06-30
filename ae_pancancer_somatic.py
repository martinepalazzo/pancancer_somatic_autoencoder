import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers import Dropout
from keras import models
from keras import layers
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization

latent_dim = 50
epchs = 50
bchs = 32
l1 = 0
l2 = 0.00002
lrte = 0.001
#fold_cv = 4
w_cv = 0.5

def ae_arch_00(encoding_dim, epochs, bachs, l1_reg, l2_reg, lrate, xtr_de,xtr_nd, xte_de,xte_nd, weight):

    earlstop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10,  mode='auto', baseline=None, restore_best_weights=False, verbose=1)
    rlrop = keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='auto')

    ################### Optimizer ###################
    optim = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)

    ################### MODEL ARCHITECTURE ###############
    # 00) INPUT
    input_de = Input(shape=(xtr_de.shape[1],))
    input_nd = Input(shape=(xtr_nd.shape[1],))

    # 01) DENSE LAYER
    encode_i_de = Dense(400, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu')(input_de)
    encode_i_nd = Dense(400, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu')(input_nd)

    # 02) BATCH NORM
    bn_i_de = BatchNormalization()(encode_i_de)
    bn_i_nd = BatchNormalization()(encode_i_nd)

    # 03) Drop Out
    #dr_i = Dropout(0.2)(bn_i)

    # 03) DENSE LAYER
    latent_de = Dense(encoding_dim, kernel_regularizer = regularizers.l2(l2_reg), activation = 'relu')(bn_i_de)
    latent_nd = Dense(encoding_dim, kernel_regularizer = regularizers.l2(l2_reg), activation = 'relu')(bn_i_nd)

    ##### shared layer #####
    shared_input = keras.layers.Concatenate(axis = -1)([latent_de, latent_nd])
    shared_output = Dense(encoding_dim, activation='relu')(shared_input)

    # 04) DECODER
    decode_i_de = Dense(400,activation='relu')(shared_output)
    decode_i_nd = Dense(400,activation='relu')(shared_output)

    # 05) DECODER
    decode_ii_de = Dense(xtr_de.shape[1],activation='sigmoid')(decode_i_de)
    decode_ii_nd = Dense(xtr_nd.shape[1],activation='sigmoid')(decode_i_nd)


    autoencoder = Model([input_de, input_nd], [decode_ii_de, decode_ii_nd])
    #autoencoder = Model(inputdim, decode_ii)
    encoder = Model([input_de, input_nd], shared_output)

    autoencoder.summary()
    
    ################## COMPILE AND FIT MODEL #############
    autoencoder.compile(optimizer=optim, loss=['binary_crossentropy','binary_crossentropy'], loss_weights=[weight, 1-weight])

    autoencoder.fit([xtr_de,xtr_nd], [xtr_de,xtr_nd],epochs=epochs, batch_size=bachs,  shuffle=True,  validation_data=([xte_de,xte_nd], [xte_de,xte_nd]), callbacks = [earlstop, rlrop])
    
    histval = autoencoder.history.history['val_loss']
    histtra = autoencoder.history.history['loss']
    
    return encoder, histtra, histval
