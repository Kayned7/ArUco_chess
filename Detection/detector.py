import cv2
import numpy as np

#cameraMatrix = np.load('calibration/camera_matrix.npy')
#distCoeffs = np.load('calibration/dist_coeffs.npy')

#lipna kalibracja bo nie mam drukarki...
cameraMatrix = np.array([[800, 0, 320],
                         [0, 800, 240],
                         [0, 0, 1]], dtype=np.float32)

distCoeffs = np.zeros((5,1), dtype=np.float32)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

marker_length = 0.06

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera erroe")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, cameraMatrix, distCoeffs
        )

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

    cv2.imshow("ArUco AR Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
