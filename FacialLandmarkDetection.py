import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
wcam, hcam = 1100, 1200
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0
NUM_FACE = 2

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms,mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

            for id,lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Test", img)
    cv2.waitKey(1)
