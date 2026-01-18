import cv2
import numpy as np
import matplotlib.pyplot as plt

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

marker_id = 13
marker_size = 300
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

cv2.imwrite(f"Markers/marker_{marker_id}.png", marker_image)
cv2.imshow(f"marker_{marker_id}",marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()