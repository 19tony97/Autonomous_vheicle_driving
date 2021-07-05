import json
import os
import matplotlib
import json

from traffic_light_detection_module.yolo import YOLO
BASE_DIR = os.path.dirname(__file__)
OUT_IMAGES_DIR = os.path.join(BASE_DIR, 'out')
from traffic_light_detection_module.predict import get_model

class TrafficLightDetector:
    
    def __init__(self, config_filepath, folder_images) -> None:
        self.config = TrafficLightDetector.load_config(config_filepath)
        self.model = get_model(self.config)
        self.folder_images = folder_images

    @staticmethod
    def load_config(config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            return config

    def predict(self, cam_image):
        return self.model.predict(cam_image)

    def save_image(self, tl_camera):
        cam_image = self.folder_images / f'camera_0.png'
        matplotlib.image.imsave(cam_image, tl_camera.data)
        return cam_image

    def detect_traffic_light(self, camera_0):
        image = self.save_image(camera_0)
        predict_traffic_light = self.predict(str(image))
        tl = len(predict_traffic_light) > 0
        
        if tl:
            # label == 1 if red light, label == 0 if green light
            tl_state = predict_traffic_light[0].label
        else:
            tl_state = None

        return tl_state