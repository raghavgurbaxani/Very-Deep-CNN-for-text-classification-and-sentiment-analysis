import tensorflow as tf
import numpy as np

num_classes=None
l2_reg_lambda=0.0005
sequence_max_length=1014
num_quantized_chars=69
embedding_size=16

l2_loss = tf.constant(0.0)

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#Function to define Bias variable
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#Function to perform convolution
def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x,W,strides=[1, 1, stride, 1], padding='SAME')
#Pooling Function
def max_pool(x, stride, k):
    return tf.nn.max_pool(x,strides=[1, stride, 1, 1], ksize=[1, k, 1, 1] , padding='SAME')
#function for convolution layer
def conv_layer(input, shape, stride):
    W=weight_variable(shape)
    b=bias_variable([shape[3]])
    return tf.nn.relu(tf.nn.conv2d(input,W,strides=[1, stride, stride, 1],padding='SAME')+b)
#Function for fully connected layer
def full_layer(input, size):
    in_size=int(input.get_shape()[1])
    W=weight_variable([in_size,size])
    b=bias_variable([size])
    
    return tf.matmul(input,W)+b


input_x = tf.placeholder(tf.int32, [None, sequence_max_length])
input_y = tf.placeholder(tf.float32, [None, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)
is_training =  tf.placeholder(tf.bool)

# Embedding Lookup 16
embedding_W = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_size], -1.0, 1.0))
embedded_characters = tf.nn.embedding_lookup(embedding_W,input_x)
embedded_characters_expanded = tf.expand_dims(embedded_characters, -1)


# First Convolutional Layer
conv_1=conv_layer(embedded_characters_expanded, shape=[3, embedding_size, 1, 64],stride=embedding_size)

#First Convolutional Block - 

conv_2=conv_layer(conv_1, shape=[3, embedding_size, 64, 64])
batch_norm_2=tf.nn.relu(tf.contrib.layers.batch_norm(conv_2))   #Batch Normalization
conv_3=conv_layer(batch_norm_2, shape=[3, embedding_size, 64, 64])
batch_norm_3=tf.nn.relu(tf.contrib.layers.batch_norm(conv_3))   #Batch Normalization
conv_4=conv_layer(batch_norm_3, shape=[3, embedding_size, 64, 64])
batch_norm_4=tf.nn.relu(tf.contrib.layers.batch_norm(conv_4))   #Batch Normalization
conv_5=conv_layer(batch_norm_4, shape=[3, embedding_size, 64, 64])
batch_norm_5=tf.nn.relu(tf.contrib.layers.batch_norm(conv_5))   #Batch Normalization
conv_6=conv_layer(batch_norm_5, shape=[3, embedding_size, 64, 128])
batch_norm_6=tf.nn.relu(tf.contrib.layers.batch_norm(conv_6))   #Batch Normalization
conv_7=conv_layer(batch_norm_6, shape=[3, embedding_size, 128, 128])
batch_norm_7=tf.nn.relu(tf.contrib.layers.batch_norm(conv_7))   #Batch Normalization
conv_8=conv_layer(batch_norm_7, shape=[3, embedding_size, 128, 128])
batch_norm_8=tf.nn.relu(tf.contrib.layers.batch_norm(conv_8))   #Batch Normalization
conv_9=conv_layer(batch_norm_8, shape=[3, embedding_size, 128, 128])
batch_norm_9=tf.nn.relu(tf.contrib.layers.batch_norm(conv_9))   #Batch Normalization
conv_10=conv_layer(batch_norm_9, shape=[3, embedding_size, 128, 256])
batch_norm_10=tf.nn.relu(tf.contrib.layers.batch_norm(conv_10))   #Batch Normalization
conv_11=conv_layer(batch_norm_10, shape=[3, embedding_size, 256, 256])
batch_norm_11=tf.nn.relu(tf.contrib.layers.batch_norm(conv_11))   #Batch Normalization

pool_1=max_pool(batch_norm_11,stride=2,k=3)

#------------------------------------------------------------------------------------------------------------

#Second Convolutional Block - 

conv_12=conv_layer(pool_1, shape=[3, embedding_size, 256, 256])
batch_norm_12=tf.nn.relu(tf.contrib.layers.batch_norm(conv_12))   #Batch Normalization
conv_13=conv_layer(pool_1, shape=[3, embedding_size, 256, 256])
batch_norm_13=tf.nn.relu(tf.contrib.layers.batch_norm(conv_13))   #Batch Normalization
conv_14=conv_layer(batch_norm_13, shape=[3, embedding_size, 256, 512])
batch_norm_14=tf.nn.relu(tf.contrib.layers.batch_norm(conv_14))   #Batch Normalization
conv_15=conv_layer(batch_norm_14, shape=[3, embedding_size, 512, 512])
batch_norm_15=tf.nn.relu(tf.contrib.layers.batch_norm(conv_15))   #Batch Normalization
conv_16=conv_layer(batch_norm_15, shape=[3, embedding_size, 512, 512])
batch_norm_16=tf.nn.relu(tf.contrib.layers.batch_norm(conv_16))   #Batch Normalization
conv_17=conv_layer(batch_norm_16, shape=[3, embedding_size, 512, 512])
batch_norm_17=tf.nn.relu(tf.contrib.layers.batch_norm(conv_17))   #Batch Normalization
conv_18=conv_layer(batch_norm_17, shape=[3, embedding_size, 512, 512])
batch_norm_18=tf.nn.relu(tf.contrib.layers.batch_norm(conv_18))   #Batch Normalization
conv_19=conv_layer(batch_norm_18, shape=[3, embedding_size, 512, 512])
batch_norm_19=tf.nn.relu(tf.contrib.layers.batch_norm(conv_19))   #Batch Normalization
conv_20=conv_layer(batch_norm_19, shape=[3, embedding_size, 512, 512])
batch_norm_20=tf.nn.relu(tf.contrib.layers.batch_norm(conv_20))   #Batch Normalization
conv_21=conv_layer(batch_norm_20, shape=[3, embedding_size, 512, 512])
batch_norm_21=tf.nn.relu(tf.contrib.layers.batch_norm(conv_21))   #Batch Normalization

pool_2=max_pool(batch_norm_21,stride=2,k=3)

#------------------------------------------------------------------------------------------------------------

#Third Convolutional Block - 

conv_22=conv_layer(pool_2, shape=[3, embedding_size, 512, 512])
batch_norm_22=tf.nn.relu(tf.contrib.layers.batch_norm(conv_22))   #Batch Normalization
conv_23=conv_layer(batch_norm_22, shape=[3, embedding_size, 512, 512])
batch_norm_23=tf.nn.relu(tf.contrib.layers.batch_norm(conv_23))   #Batch Normalization
conv_24=conv_layer(batch_norm_23, shape=[3, embedding_size, 512, 512])
batch_norm_24=tf.nn.relu(tf.contrib.layers.batch_norm(conv_24))   #Batch Normalization
conv_25=conv_layer(batch_norm_24, shape=[3, embedding_size, 512, 512])
batch_norm_25=tf.nn.relu(tf.contrib.layers.batch_norm(conv_25))   #Batch Normalization

pool_3=max_pool(batch_norm_25,stride=2,k=3)

#------------------------------------------------------------------------------------------------------------

#Fourth Convolutional Block - 

conv_26=conv_layer(pool_3, shape=[3, embedding_size, 512, 512])
batch_norm_26=tf.nn.relu(tf.contrib.layers.batch_norm(conv_26))   #Batch Normalization
conv_27=conv_layer(batch_norm_26, shape=[3, embedding_size, 512, 512])
batch_norm_27=tf.nn.relu(tf.contrib.layers.batch_norm(conv_27))   #Batch Normalization
conv_28=conv_layer(batch_norm_27, shape=[3, embedding_size, 512, 512])
batch_norm_28=tf.nn.relu(tf.contrib.layers.batch_norm(conv_28))   #Batch Normalization
conv_29=conv_layer(batch_norm_28, shape=[3, embedding_size, 512, 512])
batch_norm_29=tf.nn.relu(tf.contrib.layers.batch_norm(conv_29))   #Batch Normalization


#------------------------------------------------------------------------------------------------------------

transposed = tf.transpose(batch_norm_29, [0,3,2,1])
k_pooled = tf.nn.top_k(transposed, k=8)
reshaped = tf.reshape(k_pooled[0], [-1,8*512])

#Fully-Connected Layer 1
full_1=tf.nn.relu(full_layer(reshaped,2048))
full_drop_1=tf.nn.dropout(full_1,keep_prob=dropout_keep_prob)    #Perform dropout on fully connected layer


#Fully-Connected Layer 2
full_2=tf.nn.relu(full_layer(full_drop_1,2048))
full_drop_2=tf.nn.dropout(full_2,keep_prob=dropout_keep_prob)    #Perform dropout on fully connected layer

#Fully-Connected Layer 3
full_3=full_layer(full_drop_2,num_classes) 

#Cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= full_3,labels= input_y))

# Mask for correct predictions
correct_prediction = tf.equal(tf.argmax(full_3, 1), tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
