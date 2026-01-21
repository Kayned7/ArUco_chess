import cv2
import numpy as np

# Jeśli zrobiłeś kalibrację
cameraMatrix = np.load('Calibration/camera_matrix.npy')
distCoeffs = np.load('Calibration/dist_coeffs.npy')

# Bieda wersja bez kalibracji
""" cameraMatrix = np.array([[800, 0, 320],
                         [0, 800, 240],
                         [0, 0, 1]], dtype=np.float32)

distCoeffs = np.zeros((5,1), dtype=np.float32) """


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

marker_length = 0.04

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error")
    exit()

def draw_cube(frame, rvec, tvec, size = marker_length):
    half = size / 2
    pts = np.float32([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0],
        [-half, -half, size],
        [ half, -half, size],
        [ half,  half, size],
        [-half,  half, size]
    ])

    imgpts, _ = cv2.projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs)
    imgpts = np.int32(imgpts).reshape(-1,2)

    frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0,255,0), 2)
    frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0,0,255), 2)

    for i in range(4):
        frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i+4]), (255,0,0), 2)
    return frame

while True:
    work, frame = cap.read()
    if not work:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    x = 10          
    y = h - 10          

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, cameraMatrix, distCoeffs
        )

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            marker_id = ids[i][0]
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, marker_length/2)

            if marker_id == 0:
                frame = draw_cube(frame, rvec, tvec, size=marker_length)
            

            # marker to camera distance
            distance_to_camera = np.linalg.norm(tvec[0])
            cv2.putText(frame,
                        f"ID {ids[i][0]} Dist: {60*distance_to_camera:.2f} cm",
                        (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)  # 3 cm osie

        # distance between markers
        if len(tvecs) == 2:
            pos1 = tvecs[0][0]  # [X, Y, Z] marker 1
            pos2 = tvecs[1][0]  # marker 2

            dx, dy, dz = pos2 - pos1
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            cv2.putText(frame, f"Distance: {90*distance:.2f} cm",
            (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2)

    cv2.imshow("ArUco AR Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
