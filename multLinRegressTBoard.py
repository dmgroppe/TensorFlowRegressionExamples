""" Simple function for doing multiple linear regression with TensorFlow.
 Usage:
  python multLinRegress.py --train trainData.csv
  
 where trainData.csv is a comma delimited text file. The first column is value 
 of the output variable to be predicted. The remaining columns are the input 
 predictor variables. Do NOT include a bias predictor. This is taken care of by 
 the code.

This version of the code also visualizes the training error with Tensorboard.

 Example:
  python multLinRegress.py --train 2dLinRegExample.csv
  tensorboard --logdir=/tmp/tflow_logs/

 Code based on Jason Baldrige's softmax.py function:
  https://github.com/jasonbaldridge/try-tf

 David Groppe
 Python newbie
 Dec, 2015
 
 For more info on Tensorboard see:
 https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html
 
"""

import tensorflow.python.platform
import tensorflow as tf
import numpy as np

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
FLAGS = tf.app.flags.FLAGS



# Extract numpy representations of the outputs and features given rows consisting of:
#   output, feat_0, feat_1, ..., feat_n
def extract_data(filename):
    # Arrays to hold the outputs and feature vectors.
    outputs = []
    fvecs = []
    
    # Iterate over the rows, splitting the label from the features.
    for line in file(filename):
        row = line.split(",")
        outputs.append(float(row[0]))
        fvecs.append([float(x) for x in row[1:]])
    
    # Convert the array of float arrays into numpy float arrays.
    fvecs_np = np.matrix(fvecs).astype(np.float32)
    outputs_np = np.array(outputs).astype(np.float32)
    
    return fvecs_np,outputs_np


def norm_ftrs(npMtrx):
    # Normalize each feature to be zero mean, unit standard deviation
    nDim=npMtrx.shape
    ftrMns=np.zeros((1,nDim[1]))
    ftrSDs=np.zeros((1,nDim[1]))
    ftrMns[:1,:]=npMtrx.mean(axis=0)
    ftrSDs[:1,:]=npMtrx.std(axis=0)
    
    for a in xrange(nDim[1]):
        #npMtrx[:,a:(a+1)]=(npMtrx[:,a:(a+1)]-ftrMns[a])/ftrSDs[a]
        npMtrx[:,a:(a+1)]=(npMtrx[:,a:(a+1)]-ftrMns[0,a])/ftrSDs[0,a]
    
    return ftrMns, ftrSDs



def main(argv=None):
    
    # Get the data.
    trainDataFname = FLAGS.train
    if trainDataFname==None:
        print "Need to provide training data as argument. For example: "
        print "python multLinRegress.py --train 2dLinRegExample.csv"
        exit()
        
    print "Training Data: "
    print trainDataFname

    # Import the data from csv file
    trXtemp, trY=extract_data(trainDataFname)
    dims=trXtemp.shape
    trX=np.ones((dims[0],dims[1]))
    trX[:,:dims[1]]=trXtemp
    xMns, xSDs=norm_ftrs(trX)

    yMn=trY.mean()
    # Normalize output to zero mean
    trY=trY-yMn

    num_features=dims[1]
    train_size=dims[0]

    # create symbolic variables
    X = tf.placeholder("float", shape=[None, num_features])
    Y = tf.placeholder("float", shape=[None])

    # Regression model weights
    w = tf.Variable(tf.zeros([num_features,1]))

    # This how you can get an output of the parameters after each training run
    #w = tf.Print(w, [w], "Weights: ")

    y_model = tf.matmul(X,w) 

    cost = tf.reduce_mean((tf.pow(Y-y_model, 2))) # use mean sqr error for cost function (mean is better than sum as it makes cost independent of batch size)
    # This how you can get an output of the cost function after each training iteration
    #cost = tf.Print(cost, [cost], "cost: ") 

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # chunk below added for TensorBoard
    with tf.name_scope('test'):
        mse = tf.reduce_mean((tf.pow(Y-y_model, 2)))
        _ = tf.scalar_summary('MeanSqrError', mse)

    sess = tf.Session()
    merged = tf.merge_all_summaries() # added for TensorBoard
    writer = tf.train.SummaryWriter("/tmp/tflow_logs", sess.graph_def) # added for TensorBoard
    init = tf.initialize_all_variables()
    sess.run(init)


    # Batch training (training on all the data at once or with batch sizes larger than 1 doesn't work)
    num_epochs=2 # number of times to run through the training data
    BATCH_SIZE=1 # number of observations to train on at each weight iteration
    print "** Beginning training **"
    print "Training step: "
    for step in xrange(num_epochs * train_size // BATCH_SIZE):
        print step,
        
        offset = (step * BATCH_SIZE) % train_size
        batch_input = trX[offset:(offset + BATCH_SIZE)]
        batch_output = trY[offset:(offset + BATCH_SIZE)]
        
        # ?? pickup here to make Tensorboard work
        if step % 3 == 0:
            # Record performance for Tensorboard. I should really be doing this
            # with a hold-out test set.
            result = sess.run([merged, mse], feed_dict={X: batch_input, Y: batch_output})
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, step)
            print('MSE at step %s: %s' % (step, acc))      
        else:
            sess.run(train_op, feed_dict={X: batch_input, Y: batch_output})
    
        if offset >= train_size-BATCH_SIZE:
            print

    # Weigths from Tensor Flow model
    tWts=sess.run(w)

    # Scale wts to correct for feature normalization
    wts=tWts/xSDs.T

    # Generate bias from means of raw data
    bias=yMn-np.dot(xMns,wts)

    print
    print "Weights: ",
    print(wts) 
    print "Bias: ",
    print bias
    sess.close()
    
if __name__ == '__main__':
    tf.app.run()