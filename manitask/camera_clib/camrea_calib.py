import cv2
import numpy as np

def main():
    
    #inner coner point of the chessboard
    CHECKERBOARD = (8, 11)
    
    #length of each square (mm)
    square_size = 12.0
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  
    imgpoints = [] 

    #open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cannot open camera，check the camera or driver")
        return

    print("camera is opened, please put the chessboard in the view of camera")
    print("press 's' to save the image, press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot get the image from camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        if ret_corners:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret_corners)
            cv2.putText(frame, "Chessboard Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Chessboard Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Camera Calibration", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            if ret_corners:
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"save No. {len(objpoints)} conner points of image")
            else:
                print("no conner points detected, cannot save")
        elif key == ord('q'):
            print("ending the collection, start calibration")
            break

    cap.release()
    cv2.destroyAllWindows()

   
    # 4. calibrate camera
    if len(objpoints) < 1:
        print("no enough data for calibration")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("calibration failed, please check the data")
        return

    print("\n========== calibration result ==========")
    print("Camera intrinsic matrix\n", mtx)
    print("distortion coefficient:\n", dist)
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    print(f"\nfocal（pixle）: fx = {fx:.2f}, fy = {fy:.2f}, f = {(fx+fy)/2:.2f}")
    print("finished calibration")

if __name__ == "__main__":
    main()
