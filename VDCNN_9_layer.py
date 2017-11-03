import tensorflow as tf
import numpy as np

num_classes= None
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

pool_1=max_pool(batch_norm_3,stride=2,k=3)

#------------------------------------------------------------------------------------------------------------

#Second Convolutional Block - 
conv_4=conv_layer(pool_1, shape=[3, embedding_size, 64, 128])
batch_norm_4=tf.nn.relu(tf.contrib.layers.batch_norm(conv_4))   #Batch Normalization
conv_5=conv_layer(batch_norm_4, shape=[3, embedding_size, 128, 128])
batch_norm_5=tf.nn.relu(tf.contrib.layers.batch_norm(conv_5))   #Batch Normalization

pool_2=max_pool(batch_norm_5,stride=2,k=3)

#------------------------------------------------------------------------------------------------------------

#Third Convolutional Block - 
conv_6=conv_layer(pool_2, shape=[3, embedding_size, 128, 256])
batch_norm_6=tf.nn.relu(tf.contrib.layers.batch_norm(conv_6))   #Batch Normalization
conv_7=conv_layer(batch_norm_6, shape=[3, embedding_size, 256, 256])
batch_norm_7=tf.nn.relu(tf.contrib.layers.batch_norm(conv_7))   #Batch Normalization

pool_3=max_pool(batch_norm_7,stride=2,k=3)

#------------------------------------------------------------------------------------------------------------

#Fourth Convolutional Block - 
conv_8=conv_layer(pool_3, shape=[3, embedding_size, 256, 512])
batch_norm_8=tf.nn.relu(tf.contrib.layers.batch_norm(conv_8))   #Batch Normalization
conv_9=conv_layer(batch_norm_6, shape=[3, embedding_size, 256, 512])
batch_norm_9=tf.nn.relu(tf.contrib.layers.batch_norm(conv_9))   #Batch Normalization

#------------------------------------------------------------------------------------------------------------

transposed = tf.transpose(batch_norm_9, [0,3,2,1])
k_pooled = tf.nn.top_k(transposed, k=8)
reshaped = tf.reshape(k_pooled[0], [-1,8*512])

#Fully-Connected Layer 1
full_1=tf.nn.relu(full_layer(reshaped,2048))
full_drop_1=tf.nn.dropout(full_1,keep_prob=dropout_keep_prob)    #Perform dropout on fully connected layer

#Fully-Connected Layer 2
full_2=tf.nn.relu(full_layer(full_drop_1,2048))
full_drop_2=tf.nn.dropout(full_2,keep_prob=dropout_keep_prob)    #Perform dropout on fully connected layer

#Fully-Connected Layer 3
full_3=full_layer(full_drop_1,num_classes) 

#Cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= full_3,labels= input_y))

# Mask for correct predictions
correct_prediction = tf.equal(tf.argmax(full_3, 1), tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


