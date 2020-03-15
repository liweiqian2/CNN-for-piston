import tensorflow as tf
import os
import pandas as pd

# Dataset Parameters - CHANGE HERE
MODE = 'folder' # or 'file', if you choose a plain text file (see above).
DATASET_PATH = '215'
# Image Parameters
N_CLASSES = 1 # CHANGE HERE, total number of classes
IMG_HEIGHT = 256 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 256 # CHANGE HERE, the image width to be resized to
CHANNELS = 1 # The 3 color channels, change to 1 if grayscale


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path, mode, batch_size):
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        with open(dataset_path) as f:
            data = f.read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
       
        # List the directory
 
        classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        #for c in classes:
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            walk = os.walk(c_dir).__next__()
              # each image to the training set
            for sample in walk[2]:
               
                if sample.endswith('.png') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    sample =sample.replace(".png","")
                    label=float(sample)
                    labels.append(label)

    else:
        raise Exception("Unknown mode.")

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y

# -----------------------------------------------
# THIS IS A CLASSIC CNN 
# -----------------------------------------------
# Note that a few elements have changed (usage of queues).

# Parameters

learning_rate = 0.0001
num_steps = 5000
batch_size = 64
display_step = 5

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

# Build the data input
X, Y = read_images(DATASET_PATH, MODE, batch_size)
#X1,Y1 represent the vailidation data sets
#X1, Y1 = read_images(DATASET_PATH1, MODE, batch_size)

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        #1: Convolution Layer with 64 filters and a kernel size of 3
        conv1 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
        #2: Convolution Layer with 64 filters and a kernel size of 3
        conv1 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        #3: Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)



        #4: Convolution Layer with 128 filters and a kernel size of 3
        conv4 = tf.layers.conv2d(conv1, 128, 3, activation=tf.nn.relu)
        #5: Convolution Layer with 128 filters and a kernel size of 3
        conv5 = tf.layers.conv2d(conv4, 128, 3, activation=tf.nn.relu)
        #6: Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv6 = tf.layers.max_pooling2d(conv5, 2, 2)


        #7: Convolution Layer with 256 filters and a kernel size of 3
        conv7 = tf.layers.conv2d(conv6, 256, 3, activation=tf.nn.relu)
        #8: Convolution Layer with 256 filters and a kernel size of 3
        conv8 = tf.layers.conv2d(conv7, 256, 3, activation=tf.nn.relu)
        #9: Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv9 = tf.layers.max_pooling2d(conv8, 2, 2)


        #10: Convolution Layer with 512 filters and a kernel size of 5
        conv10 = tf.layers.conv2d(conv9, 512, 3, activation=tf.nn.relu)
        #11: Convolution Layer with 512 filters and a kernel size of 5
        conv11 = tf.layers.conv2d(conv10, 512, 3, activation=tf.nn.relu)
        #12: Convolution Layer with 512 filters and a kernel size of 5
        conv12 = tf.layers.conv2d(conv11, 512, 3, activation=tf.nn.relu)
        #13: Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv13 = tf.layers.max_pooling2d(conv12, 2, 2)

        
        #14: Convolution Layer with 512 filters and a kernel size of 5
        conv14 = tf.layers.conv2d(conv13, 512, 3, activation=tf.nn.relu)
        #15: Convolution Layer with 512 filters and a kernel size of 5
        conv15 = tf.layers.conv2d(conv14, 512, 3, activation=tf.nn.relu)
        #16:Convolution Layer with 512 filters and a kernel size of 5
        conv16 = tf.layers.conv2d(conv15, 512, 3, activation=tf.nn.relu)
        #17: Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv17 = tf.layers.max_pooling2d(conv16, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv17)

        #18: Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        out = tf.reshape(out,[64,])

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)
#logits_test = conv_net(X1, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with test logits)
loss_op = tf.losses.mean_squared_error(labels=Y,predictions=logits_test)
#loss_op1 = tf.losses.mean_squared_error(labels=Y1,predictions=logits_test)
# Define loss and optimizer (with train logits, for dropout to take effect)
#loss_op = tf.losses.mean_squared_error(labels=Y,predictions=logits_train)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()

# Start training or testing
with tf.Session() as sess:


    # Run the initializer
    sess.run(init)

    #saver.restore(sess, 'my_tf_model/')
    #print("Model restored from file: " )

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss
            #_, loss= sess.run([train_op, loss_op])
            #_, loss,loss1= sess.run([train_op, loss_op,loss_op1])

            # calculate batch loss for test
            loss,a,b= sess.run([loss_op,logits_test,Y])

            # calculate correlation coefficient for test
            #s1=pd.Series(a) 
            #s2=pd.Series(b)
            #corr=s1.corr(s2) 
            #print(corr)

            #file_handle=open('corr.txt',mode='a+')
            #file_handle.write("{:.4f}".format(corr)+"\n")
            #file_handle.close()

            #file_handle=open('predict.txt',mode='a+')
            #for i in range(len(a)):
                #s = str(a[i]).replace('[','').replace(']','')
                #s = s.replace("'",'').replace(',','') +'\n'  
                #file_handle.write(s)
            #file_handle.write("end\n")
            #file_handle.close()

            #file_handle=open('actual.txt',mode='a+')
            #for i in range(len(b)):
                #s1 = str(b[i]).replace('[','').replace(']','')
                #s1 = s1.replace("'",'').replace(',','') +'\n'  
                #file_handle.write(s1)
            #file_handle.write("end\n")
            #file_handle.close()

            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss))
            #print("Step " + str(step) + ", Minibatch Loss= " +     "{:.4f}".format(loss)     +  ", Minibatch Loss1= "     +  "{:.4f}".format(loss1))
            file_handle=open('1.txt',mode='a+')
            file_handle.write("{:.4f}".format(loss)+"\n")
            file_handle.close()
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)
    
    print("Optimization Finished!")

    # Save your model
    saver.save(sess, 'my_tf_model/')





