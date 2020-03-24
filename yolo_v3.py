import tensorflow as tf


INPUT_SIZE = 416

ANCHORS = [(10, 13), (16, 30), (33, 23),
           (30, 61), (62, 45), (59, 119),
           (116, 90), (156, 198), (373, 326)]

ANCHORS_TINY = [(10, 14),  (23, 27),  (37, 58),
                (81, 82),  (135, 169),  (344, 319)]

slim = tf.contrib.slim


def darknet53(inputs):
    net = conv_op(inputs, 32, 3, 1)
    net = conv_op(net, 64, 3, 2)
    net = darknet53_block(net, 32)
    net = conv_op(net, 128, 3, 2)
    for i in range(2):
        net = darknet53_block(net, 64)

    net = conv_op(net, 256, 3, 2)
    for i in range(8):
        net = darknet53_block(net, 128)
    route_1 = net

    net = conv_op(net, 512, 3, 2)
    for i in range(8):
        net = darknet53_block(net, 256)
    route_2 = net

    net = conv_op(net, 1024, 3, 2)
    for i in range(4):
        net = darknet53_block(net, 512)
    outputs = net

    return route_1, route_2, outputs


def conv_op(inputs, num_filters, kernel_size, strides):
    if strides > 1:
        inputs = tf.pad(inputs, [[0, 0], [kernel_size // 2, kernel_size // 2],
                                 [kernel_size // 2, kernel_size // 2], [0, 0]])
        outputs = slim.conv2d(inputs, num_filters, kernel_size, stride=strides, padding='VALID')
    else:
        outputs = slim.conv2d(inputs, num_filters, kernel_size, stride=strides, padding='SAME')
    return outputs


def darknet53_block(inputs, filters):
    net = conv_op(inputs, filters, 1, 1)
    net = conv_op(net, filters * 2, 3, 1)
    outputs = net + inputs
    return outputs


def spp_block(inputs):
    return tf.concat([slim.max_pool2d(inputs, 13, 1, 'SAME'),
                      slim.max_pool2d(inputs, 9, 1, 'SAME'),
                      slim.max_pool2d(inputs, 5, 1, 'SAME'),
                      inputs], axis=3)


def yolo_block(inputs, filters, with_spp=False):
    net = conv_op(inputs, filters, 1, 1)
    net = conv_op(net, filters * 2, 3, 1)
    net = conv_op(net, filters, 1, 1)

    if with_spp:
        net = spp_block(net)
        net = conv_op(net, filters, 1, 1)

    net = conv_op(net, filters * 2, 3, 1)
    net = conv_op(net, filters, 1, 1)
    route = net
    outputs = conv_op(net, filters * 2, 3, 1)
    return route, outputs


def detect_op(inputs, num_classes, anchors, grid_size):
    num_anchors = len(anchors)
    predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1,
                              normalizer_fn=None, activation_fn=None,
                              biases_initializer=tf.zeros_initializer())

    predictions = tf.reshape(predictions, [-1, num_anchors * grid_size * grid_size, 5 + num_classes])
    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)

    stride = (INPUT_SIZE // grid_size, INPUT_SIZE // grid_size)

    grid_x = tf.range(grid_size, dtype=tf.float32)
    grid_y = tf.range(grid_size, dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = tf.sigmoid(box_centers)
    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
    anchors = tf.tile(anchors, [grid_size * grid_size, 1])

    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    confidence = tf.sigmoid(confidence)
    classes = tf.nn.sigmoid(classes)

    predictions = tf.concat([box_centers, box_sizes, confidence, classes], axis=-1)
    return predictions


def convert_result(prediction):
    scores = tf.expand_dims(prediction[0][:, 4], 1) * prediction[0][:, 5:]
    boxes = prediction[0][:, :4]

    # find each box class, only select the max score
    box_classes = tf.argmax(scores, axis=1)
    box_class_scores = tf.reduce_max(scores, axis=1)

    # filter the boxes by the score threshold
    filter_mask = box_class_scores >= 0.6
    scores = tf.boolean_mask(box_class_scores, filter_mask)
    boxes = tf.boolean_mask(boxes, filter_mask)
    box_classes = tf.boolean_mask(box_classes, filter_mask)

    # non max suppression (do not distinguish different classes)
    # box (x, y, w, h) -> _box (x1, y1, x2, y2)
    _boxes = tf.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                       boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)
    nms_indices = tf.image.non_max_suppression(_boxes, scores, 10, 0.6)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    box_classes = tf.gather(box_classes, nms_indices)
    return scores, boxes, box_classes


def yolo_v3(inputs, num_classes, is_training=False, reuse=False, with_spp=False):

    batch_norm_params = {'decay': 0.9,
                         'epsilon': 1e-05,
                         'scale': True,
                         'is_training': is_training,
                         'fused': None}

    with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            biases_initializer=None,
                            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
            with tf.variable_scope('darknet-53'):
                route_1, route_2, net = darknet53(inputs)

            with tf.variable_scope('yolo-v3'):
                net, output_1 = yolo_block(net, 512, with_spp)
                detect_1 = detect_op(output_1, num_classes, ANCHORS[6:9], 13)

                net = conv_op(net, 256, 1, 1)
                upsample_size = route_2.get_shape().as_list()
                net = tf.image.resize_nearest_neighbor(net, (upsample_size[1], upsample_size[2]))
                net = tf.concat([net, route_2], axis=3)

                net, output_2 = yolo_block(net, 256)
                detect_2 = detect_op(output_2, num_classes, ANCHORS[3:6], 26)

                net = conv_op(net, 128, 1, 1)
                upsample_size = route_1.get_shape().as_list()
                net = tf.image.resize_nearest_neighbor(net, (upsample_size[1], upsample_size[2]))
                net = tf.concat([net, route_1], axis=3)

                _, output_3 = yolo_block(net, 128)
                detect_3 = detect_op(output_3, num_classes, ANCHORS[0:3], 52)

                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                scores, boxes, box_classes = convert_result(detections)
                return scores, boxes, box_classes


def yolo_v3_spp(inputs, num_classes, is_training=False, reuse=False):
    return yolo_v3(inputs, num_classes, is_training=is_training, reuse=reuse, with_spp=True)


def yolo_v3_tiny(inputs, num_classes, is_training=False, reuse=False):

    batch_norm_params = {'decay': 0.9,
                         'epsilon': 1e-05,
                         'scale': True,
                         'is_training': is_training,
                         'fused': None}

    with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.max_pool2d]):
        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                with tf.variable_scope('yolo-v3-tiny'):
                    net = conv_op(inputs, 16, 3, 1)
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = conv_op(net, 32, 3, 1)
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = conv_op(net, 64, 3, 1)
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = conv_op(net, 128, 3, 1)
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')

                    net = conv_op(net, 256, 3, 1)
                    route_1 = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')

                    net = conv_op(net, 512, 3, 1)
                    net = slim.max_pool2d(net, [2, 2], stride=1, padding="SAME", scope='pool2')
                    net = conv_op(net, 1024, 3, 1)
                    net = conv_op(net, 256, 1, 1)
                    output_1 = net

                    output_1 = conv_op(output_1, 512, 3, 1)
                    detect_1 = detect_op(output_1, num_classes, ANCHORS_TINY[3:6], 13)

                    net = conv_op(net, 128, 1, 1)
                    upsample_size = route_1.get_shape().as_list()
                    net = tf.image.resize_nearest_neighbor(net, (upsample_size[1], upsample_size[2]))
                    net = tf.concat([net, route_1], axis=3)

                    output_2 = conv_op(net, 256, 3, 1)
                    detect_2 = detect_op(output_2, num_classes, ANCHORS_TINY[0:3], 26)

                    detections = tf.concat([detect_1, detect_2], axis=1)
                    scores, boxes, box_classes = convert_result(detections)
                    return scores, boxes, box_classes