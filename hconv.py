"""
Convolution process

Copyright (C) 2019 Leonardo Ledesma Domínguez

This process receives two inputs: the HoFs of each layer and Feature Maps
        FM_NEW (batch, C_out, h_conv, w_conv) = sum(HoFs(c_out,c_in,H,Z)  * FM [batch, c_in, h_ant,w_ant]) by c_out

the size of batch is given in the process of implict way.

"""
import tensorflow as tf

def HermiteConv(featureMap,hofs, strides, padding, bias = False):

    #info general

    hofs_shape = hofs.shape
    C_OUT = hofs_shape[0]
    C_IN = hofs_shape[1]
    size = hofs_shape[2]
    featureMap_shape = featureMap.shape
    print(featureMap_shape)
    size_batch = featureMap_shape[0]
    h = featureMap_shape[2]
    w = featureMap_shape[3]
    conv_start = 0

    print(' ', type(featureMap_shape))
    for k in range(size_batch):
        starting = 0
        for j in range(C_OUT):
            flag = 0
            for i in range (C_IN):
                #print(fm[i,:,:],h,w)
                kernel = tf.reshape(hofs[j,i,:,:],[size, size, 1,1], name = 'kernel')
                #print(fm[i,:,:].shape, h, w, C_IN, C_OUT)
                fm = tf.reshape(featureMap[k,i,:,:],[1,h,w,1], name = 'fm')
                conv = tf.nn.conv2d(fm,kernel,strides = strides ,padding=padding, use_cudnn_on_gpu=True )
                if flag == 0:
                    suma = conv
                    flag = 1
                else:
                    suma += conv

            suma = tf.reshape(suma, [1,h,w])

            if starting == 0:
                fm_new = suma
                starting = 1
            else:
                fm_new = tf.concat([fm_new, suma], axis=0)

        fm_new = tf.reshape(fm_new, [1,C_OUT,h,w])
        if conv_start == 0:
            fm_new_total = fm_new
            conv_start = 1
        else:
            fm_new_total = tf.concat([fm_new_total, fm_new], axis = 0)

    return fm_new_total

def main():
    import matplotlib.pyplot as plt
    tf.enable_eager_execution()
    from test import convTensores
    fm, hofs = convTensores.examples()
    print(fm.shape)
    print(hofs.shape)
    #fm = fm[1,:,:,:]
    fm_new = HermiteConv(fm,hofs,strides=[1,1,1,1],padding='SAME')
    #print(fm_new[0,1,:,:])

    #print(fm_new[0,0,:,:])
    plt.gray()
    plt.figure(1)
    plt.imshow(fm[0,0,:,:])
    plt.figure(2)
    plt.imshow(fm_new[0,0,:,:])
    plt.show()


if __name__  == '__main__':
    main()

