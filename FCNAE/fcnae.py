import tensorflow as tf

class FCNAE:
    def __init__(self, config):
        self.n_channel = config.n_channel
        self.stride = config.stride
        self.strides = [self.stride, self.stride]
        self.n_layers = config.n_layers
        self.weights = {}
        self.X = tf.placeholder(tf.float32, shape=[None,None,None,self.n_channel])

    def _conv2d_layer(self, inputs, filters_size, strides=[1, 1], name=None, padding="SAME", activation=None):
        filters = self._get_conv_filters(filters_size, name)
        strides = [1, *strides, 1]

        conv_layer = tf.nn.conv2d(inputs, filters, strides=strides, padding=padding, name=name + "layer")

        if activation != None:
            conv_layer = activation(conv_layer)

        return conv_layer

    def _conv2d_transpose(self, inputs, transpose_filters_shape, strides=[1, 1], output_shape=None, name=None, padding="SAME", activation=None):
        '''
        :param inputs: [batch, height, width, in_channels]
        :param transpose_filters_shape: [batch_size, output_height, output_width, output_channels]
        '''

        inputs_shape = tf.shape(inputs)
        batch_size, inputs_height, inputs_width, input_channels = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]
        filters_height, filters_width, output_channels, input_channels = transpose_filters_shape

        if output_shape == None:
            output_size = tf.stack(
                self._calc_output_size([inputs_height, inputs_width], [filters_height, filters_width], strides, padding))  # tf.pack renamed tf.stack
            output_shape = [batch_size, *output_size, output_channels]

        filters = self._get_conv_filters(transpose_filters_shape, name)
        strides = [1, *strides, 1]

        conv_transpose = tf.nn.conv2d_transpose(inputs, filters, output_shape=output_shape, strides=strides, padding=padding, name=name)

        if activation != None:
            conv_transpose = activation(conv_transpose)

        return conv_transpose

    def _get_conv_filters(self, filters_size, name):
        initializer = tf.contrib.layers.xavier_initializer()
        conv_weights = tf.Variable(initializer(filters_size), name=name + "weights")

        self.weights[name] = conv_weights
        return conv_weights

    def _calc_output_size(self, inputs_size, filters_size, strides, padding):  # For conv_transpose
        inputs_height, inputs_width = inputs_size
        filters_height, filters_width = filters_size
        strides_height, strides_width = strides

        if padding == "SAME":
            output_height = inputs_height * strides_height
            output_width = inputs_width * strides_width

        else:  # padding="VALID"
            output_height = (inputs_height - 1) * strides_height + filters_height
            output_width = (inputs_width - 1) * strides_width + filters_width

        return [output_height, output_width]

    def neuralnet(self):
        scale = (self.stride**2) - 1 # If stride is N, then the multiple of the number of channels is (stride**N - 1)
        input_channels_list = [(self.n_channel * (scale**i)) for i in range(0, self.n_layers)]
        #print(input_channels_list)
        output_channels_list = [(self.n_channel * (scale**i)) for i in range(1, self.n_layers + 1)]
        #print(output_channels_list)
        inputs_shape_list = []

        '''
       Encoder. The number of layers is same 'n_layer'.
       '''
        encoder = self.X
        #print(encoder.shape)
        for i in range(self.n_layers):
            in_channels = input_channels_list[i]
            output_channels = output_channels_list[i]
            inputs_shape = tf.shape(encoder)
            inputs_shape_list.append(inputs_shape)
            encoder = self._conv2d_layer(encoder,
                                         filters_size=[6,6,in_channels, output_channels],
                                         strides=self.strides,
                                         name="encoder_{}".format(i),
                                         padding="SAME",
                                         activation=tf.nn.relu)
            #print(encoder.shape)

        '''
       Decoder. The number of layers is same 'n_layer'
       '''
        decoder = encoder
        for i in range(self.n_layers):
            in_channels = input_channels_list[-1-i]
            output_channels = output_channels_list[-1-i]
            inputs_shape = inputs_shape_list[-1-i]

            decoder = self._conv2d_transpose(decoder,
                                             transpose_filters_shape=[6,6,in_channels,output_channels],
                                             output_shape =inputs_shape,
                                             strides=self.strides,
                                             name="decoder_{}".format(i),
                                             padding="SAME",
                                             activation=tf.nn.relu)
            #print(decoder.shape)

        self.output = self._conv2d_layer(decoder, filters_size=[3, 3, self.n_channel, self.n_channel], padding="SAME", name="reconstruction", activation=tf.nn.sigmoid)

    def optimize(self, config):
        self.learning_rate = config.learning_rate
        self.beta_1 = config.beta_1
        self.beta_2 = config.beta_2
        self.epsilon = config.epsilon

        self.cost = tf.reduce_mean(tf.pow(self.output - self.X, 2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1,
                                                beta2=self.beta_2, epsilon=self.epsilon).minimize(self.cost)

    def summary(self):
        '''
        for weight in list(self.weights.keys()):
            tf.summary.histogram(weight, self.weights[weight])
        for bias in list(self.biases.keys()):
            tf.summary.histogram(bias, self.biases[bias])
        '''

        tf.summary.scalar('Loss', self.cost)
        tf.summary.scalar('Learning rate', self.learning_rate)

        self.summaries = tf.summary.merge_all()
