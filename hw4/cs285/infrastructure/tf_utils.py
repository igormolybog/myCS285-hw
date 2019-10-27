import tensorflow as tf
import os
############################################
############################################


def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):

    # DONE: GETTHIS from HW1
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    output_placeholder = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            output_placeholder = tf.layers.dense(inputs=output_placeholder, units=size, activation=activation) # DONE HINT: use tf.layers.dense (specify <input>, <size>, activation=<?>)
        output_placeholder = tf.layers.dense(inputs=output_placeholder, units=output_size, activation=output_activation) # DONE HINT: use tf.layers.dense (specify <input>, <size>, activation=<?>)
    return output_placeholder


############################################
############################################


def create_tf_session(use_gpu, gpu_frac=0.6, allow_gpu_growth=True, which_gpu=0):
    if use_gpu:
        # gpu options
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_frac,
            allow_growth=allow_gpu_growth)
        # TF config
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
            allow_soft_placement=True,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        # set env variable to specify which gpu to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    else:
        # TF config without gpu
        config = tf.ConfigProto(device_count={'GPU': 0})

    # use config to create TF session
    sess = tf.Session(config=config)
    return sess

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

from sys import stdout

def watched(self, tensors_gen=lambda x: [x],
                    string_gen = lambda x: "WATCH: "+str(x.name)+": ",
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
