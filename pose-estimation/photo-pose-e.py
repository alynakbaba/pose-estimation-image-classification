import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
img = cv2.imread("veriseti/test/Downdog/00000048.jpg")

img = cv2.resize(img, (600, 400))
result = pose.process(img)

mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                        mp_draw.DrawingSpec((255, 0, 255), 2, 2))
cv2.imshow("Pose Estimation", img)

h, w, c = img.shape
opImg = np.zeros([h, w, c])
opImg.fill(255)
mp_draw.draw_landmarks(opImg, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                        mp_draw.DrawingSpec((255, 0, 255), 2, 2))
cv2.imshow("Extracted Pose", opImg)

print(result.pose_landmarks)

cv2.waitKey(0)
cv2.destroyAllWindows()
