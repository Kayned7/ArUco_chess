import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.01

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []   
imgpoints = []   

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

images = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))


print(f"Detected images: {len(images)}")
if len(images) == 0:
    raise RuntimeError("Images not found")

image_size = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Couldn't find: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    print(f"{os.path.basename(fname)} -> {'OK' if ret else 'ERROR'}")

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow("Calibration", img)
    cv2.waitKey(300)

cv2.destroyAllWindows()

if len(objpoints) == 0:
    raise RuntimeError("No Chessboards found")

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("\n=== Error per image ===")

total_error = 0
errors = []

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        cameraMatrix,
        distCoeffs
    )

    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    errors.append(error)
    total_error += error

    print(f"Image {i:02d} | error = {error:.3f} px")

mean_error = total_error / len(objpoints)
print(f"\nAVG error: {mean_error:.3f} px")

print("\n=== Calibration results ===")
print("RMS error (OK<0.5):", ret)
print("Camera matrix:\n", cameraMatrix)
print("Distortion coefficients:\n", distCoeffs)


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
print("Saving to:", os.path.join(OUT_DIR, "camera_matrix.npy"))

np.save(os.path.join(OUT_DIR, "camera_matrix.npy"), cameraMatrix)
np.save(os.path.join(OUT_DIR, "dist_coeffs.npy"), distCoeffs)

