"""
Modulation process

Copyright (C) 2019 Leonardo Ledesma Domínguez

This process receives two inputs: the Learned Filters of each layer and Hermite Filter Bank
        HB (U,H,Z)  ∘ LF [COUT, CIN, U, H,Z]
With the product element by element Hadamard, orientation by orientation

"""
import tensorflow as tf
from hcn.hermite import getHermiteFilterBank

def generateHoFs(LF,Ori,size, scale = 1):
    #tf.enable_eager_execution()
    HB = getHermiteFilterBank(scale, Ori, size, size)
    LF_shape = LF.shape
    C_OUT = LF_shape[0]
    C_INT = LF_shape[1]
    #hofs = np.zeros([C_OUT*Ori, C_INT*Ori, size, size])
    flag = 0

    for cout in range(C_OUT):
            for j in range (Ori):
                prod = tf.multiply(LF[cout, :, :, :, :], HB[j,:,:])
                print(prod.shape)
                prod = tf.reshape(prod, [C_INT*Ori,size,size])
                prod = tf.reshape(prod, [1, C_INT*Ori,size,size])
                #assigments
                if flag == 0:
                    hofs = prod
                    flag =1
                else:
                    hofs = tf.concat([hofs,prod], axis=0)
    return hofs

def main():
    tf.enable_eager_execution()
    from test import tensores
    LF,LF2 = tensores.examples()
    print(LF2.shape)
    LF2 = tf.cast(LF2, tf.float32)
    A = generateHoFs(LF2,4,5)
    print (A.shape)
    print(A[:,:,:,:])

if __name__  == '__main__':
    main()


