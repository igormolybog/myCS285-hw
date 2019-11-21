import tensorflow as tf

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass)

        Hint: use tf.layers.dense
    """
    output_placeholder = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            output_placeholder = tf.layers.dense(output_placeholder, size, activation=activation)
        output_placeholder = tf.layers.dense(output_placeholder, output_size, activation=output_activation)
    return output_placeholder

from sys import stdout

def watched(self, tensors_gen=lambda x: [x, tf.shape(x)],
                    string_gen = lambda x: "\n\n\n\n\nWATCH: "+str(x.name)+": ",
                    output_stream=stdout):
    """
        attaches a printing operator to the tensor self

        param: extra_processor(t: tf.Tensor, s: stream) -> tf.operation
        is a function that must return a print operation
    """
    print_op = tf.print(string_gen(self), *tensors_gen(self), output_stream=output_stream)
    return tf.tuple([self], control_inputs=[print_op])[0]

def tf_debug():
    tf.Tensor.watched = watched
