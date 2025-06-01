import cv2
import numpy as np
import os

# === 설정 ===
CHECKERBOARD = (8, 6)  # 내부 코너 개수 (가로, 세로)
SAVE_DIR = 'calib_images'  # 저장 폴더
os.makedirs(SAVE_DIR, exist_ok=True)
square_size_mm = 25  # 체커보드 한 칸 실제 크기 (mm)

# === 웹캠 열기 ===
cap = cv2.VideoCapture(0)  # 0번 카메라

img_count = 0
print("스페이스바: 이미지 저장, ESC: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()
    if ret_cb:
        # 코너 그리기
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret_cb)
        cv2.putText(display, "Corners detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(display, "No corners", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Checkerboard Live', display)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32 and ret_cb:  # 스페이스바
        fname = os.path.join(SAVE_DIR, f'calib_{img_count:02d}.jpg')
        cv2.imwrite(fname, frame)
        print(f"저장됨: {fname}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()

import glob

# === 3D 체커보드 좌표 생성 ===
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size_mm

objpoints = []  # 3D 점 (월드 좌표)
imgpoints = []  # 2D 점 (이미지 좌표)

# === 이미지에서 코너 검출 ===
images = glob.glob(f'{SAVE_DIR}/*.jpg')
print(f"총 {len(images)}장 이미지에서 코너 검출 시도")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
    else:
        print(f"코너 검출 실패: {fname}")

cv2.destroyAllWindows()

# === 카메라 캘리브레이션 ===
if len(objpoints) < 10:
    print("코너가 검출된 이미지가 10장 미만입니다. 더 많이 촬영하세요!")
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== 캘리브레이션 결과 ===")
print("재투영 오차(RMS):", ret)
print("카메라 행렬(K):\n", mtx)
print("왜곡 계수(dist):", dist.ravel())

# === 결과 저장 ===
np.savez('calib_result.npz', K=mtx, dist=dist)
print("\n캘리브레이션 결과가 calib_result.npz에 저장되었습니다.")

# === 왜곡 보정 예시 ===
test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
cv2.imshow('Original vs Undistorted', np.hstack((test_img, undistorted)))
cv2.waitKey(0)
cv2.destroyAllWindows()