# real-time_object_detection_YOLOv3
Real-time object detection using COCO-pretrained YOLOv3 model
### Environment

- python==3.6.5
- tensorflow==1.12.0
- opencv-python


### Getting Started

1. Download the required model's .weights files and place in the "weights" folder.

   | Model                                                        | Train         | Test     | mAP  | FLOPS     |
   | ------------------------------------------------------------ | ------------- | -------- | ---- | --------- |
   | [YOLOv3](https://pjreddie.com/media/files/yolov3.weights)    | COCO trainval | test-dev | 55.3 | 65.86 Bn  |
   | [YOLOv3-spp](https://pjreddie.com/media/files/yolov3-spp.weights) | COCO trainval | test-dev | 60.6 | 141.45 Bn |
   | [YOLOv3-tiny](https://pjreddie.com/media/files/yolov3-tiny.weights) | COCO trainval | test-dev | 33.1 | 5.56 Bn   |

2. run "convert_weights" to convert Darknet's .weights files into tensorflow's .ckpt files.

3. run "realtime_detection.py" to detect from image, video or camera.

