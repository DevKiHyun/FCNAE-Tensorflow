import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

def training(FCNAE, config):
    '''
     SETTING HYPERPARAMETER
     '''
    training_epoch = config.training_epoch
    batch_size = config.batch_size
    n_data = mnist.train.num_examples
    total_batch = int(mnist.train.num_examples / batch_size)
    total_iteration = training_epoch * total_batch
    n_iteration = 0

    # Build Network
    FCNAE.neuralnet()
    # Optimize Network
    FCNAE.optimize(config)
    '''
    # Summary of VAE Network
    FCNAE.summary()
    '''

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("Total the number of Data : " + str(n_data))
    print("Total Step per 1 Epoch: {}".format(total_batch))
    print("The number of Iteration: {}".format(total_iteration))

    for epoch in range(training_epoch):
        avg_cost = 0
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            batch_xs = np.reshape(batch_xs, (batch_size, 28, 28, 1))

            _cost, _ = sess.run([FCNAE.cost, FCNAE.optimizer], feed_dict={FCNAE.X: batch_xs})
            avg_cost += _cost / total_batch

        if epoch % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    n_sample = 8
    sample_original_images = mnist.test.images[:n_sample].reshape(-1, 28, 28, 1)
    result_images = sess.run(FCNAE.output, feed_dict={FCNAE.X: sample_original_images})

    images_list = [sample_original_images, result_images]

    columns = 8
    rows = 2
    fig, axis = plt.subplots(rows, columns)

    for i in range(columns):
        for j in range(rows):
            axis[j, i].imshow(images_list[j][i].reshape(28, 28))

    plt.show()