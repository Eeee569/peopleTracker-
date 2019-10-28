


import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from src.yolo import YOLO

VIDEO_PATH = "./testing/Screencast.mp4"





if __name__ == '__main__':

    yolo = YOLO()

    vid = cv2.VideoCapture(VIDEO_PATH)
    while True:
        return_value,frame = vid.read()
        image = Image.fromarray(frame)

        yolo.detect_image(image)



        result = np.asarray(image)



        cv2.imshow("result",frame)







        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
