from yolo_v3 import *
import tensorflow as tf
import numpy as np


NUM_CLASSES = 80


def convert(model, weights_file, ckpt_file):
    f = open(weights_file, "rb")
    weights = np.fromfile(f, dtype=np.float32)

    tf.reset_default_graph()
    with tf.variable_scope('detector'):
        inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        
        _ = model(inputs, NUM_CLASSES)
        var_list = tf.global_variables(scope='detector')

        ptr = 5
        i = 0
        assign_ops = []

        while i < len(var_list) - 1:
            var1 = var_list[i]
            var2 = var_list[i + 1]

            if 'Conv' in var1.name.split('/')[-2]:
                if 'BatchNorm' in var2.name.split('/')[-2]:
                    gamma, beta, mean, var = var_list[i + 1:i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[ptr:ptr + num_params].reshape(shape)
                        ptr += num_params
                        assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                    i += 4

                elif 'Conv' in var2.name.split('/')[-2]:
                    bias = var2
                    bias_shape = bias.shape.as_list()
                    bias_params = np.prod(bias_shape)
                    bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                    ptr += bias_params
                    assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                    i += 1

                shape = var1.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                i += 1

    saver = tf.train.Saver(tf.global_variables(scope='detector'))

    with tf.Session() as sess:
        sess.run(assign_ops)
        saver.save(sess, save_path=ckpt_file)


if __name__ == '__main__':
    convert(yolo_v3, './weights/yolov3.weights', './yolov3/model.ckpt')
    convert(yolo_v3_spp, './weights/yolov3-spp.weights', './yolov3-spp/model.ckpt')
    convert(yolo_v3_tiny, './weights/yolov3-tiny.weights', './yolov3-tiny/model.ckpt')
