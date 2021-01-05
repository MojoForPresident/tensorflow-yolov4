import tensorflow as tf
tf.config.run_functions_eagerly(True)

from py_src.yolov4.tf import YOLOv4

# yolo = YOLOv4()
yolo = YOLOv4(tiny=True)

yolo.classes = "test/dataset/coco.names"
yolo.input_size = (640, 480)

yolo.make_model()
# yolo.load_weights("test/yolov4.weights", weights_type="yolo")
yolo.load_weights("test/yolov4-tiny.weights", weights_type="yolo")

yolo.inference(media_path="test/kite.jpg")
# yolo.inference(media_path="test/test.mp4", is_image=False)
