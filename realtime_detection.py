from yolo_v3 import *
import tensorflow as tf
import time
import cv2


INPUT_SIZE = 416


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id_, name in enumerate(f):
            names[id_] = name.split('\n')[0]
    return names


def detect_from_image(image_path, model, model_file):

    classes = load_coco_names('coco.names')

    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 3])
    with tf.variable_scope('detector'):
        scores, boxes, box_classes = model(inputs, len(classes))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_file)

        img = cv2.imread(image_path)
        img_h, img_w, _ = img.shape
        img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_in = img_rgb.reshape((1, INPUT_SIZE, INPUT_SIZE, 3)) / 255.

        scores, boxes, box_classes = sess.run([scores, boxes, box_classes], feed_dict={inputs: img_in})

    for i in range(len(scores)):
        box_class = classes[box_classes[i]]

        left = int((boxes[i, 0] - boxes[i, 2] / 2) * img_w / INPUT_SIZE)
        right = int((boxes[i, 0] + boxes[i, 2] / 2) * img_w / INPUT_SIZE)
        top = int((boxes[i, 1] - boxes[i, 3] / 2) * img_h / INPUT_SIZE)
        bottom = int((boxes[i, 1] + boxes[i, 3] / 2) * img_h / INPUT_SIZE)

        score = scores[i]

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(img, (left, top - 20), (right, top), (125, 125, 125), -1)
        cv2.putText(img, box_class + ': %.2f' % score, (left + 5, top - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow('YOLOv3 result', img)
    cv2.waitKey()


def detect_from_video(video_path, model, model_file):

    classes = load_coco_names('coco.names')

    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 3])
    with tf.variable_scope('detector'):
        scores_, boxes_, box_classes_ = model(inputs, len(classes))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_file)

        cap = cv2.VideoCapture(video_path)
        while True:
            timer = time.time()
            _, frame = cap.read()

            img_h, img_w, _ = frame.shape
            img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_in = img_rgb.reshape((1, INPUT_SIZE, INPUT_SIZE, 3)) / 255.
            scores, boxes, box_classes = sess.run([scores_, boxes_, box_classes_], feed_dict={inputs: img_in})

            if scores is not None:
                for i in range(len(scores)):

                    box_class = classes[box_classes[i]]

                    left = int((boxes[i, 0] - boxes[i, 2] / 2) * img_w / INPUT_SIZE)
                    right = int((boxes[i, 0] + boxes[i, 2] / 2) * img_w / INPUT_SIZE)
                    top = int((boxes[i, 1] - boxes[i, 3] / 2) * img_h / INPUT_SIZE)
                    bottom = int((boxes[i, 1] + boxes[i, 3] / 2) * img_h / INPUT_SIZE)

                    score = scores[i]

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, top - 20), (right, top), (125, 125, 125), -1)
                    cv2.putText(frame, box_class + ' : %.2f' % score, (left + 5, top - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            print('Inference Time: %.4fs' % (time.time() - timer))
            cv2.imshow('YOLOv3 result', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    # detect_from_image('test.jpg', yolo_v3, './yolov3/model.ckpt')
    detect_from_video(1, yolo_v3, './yolov3/model.ckpt')
