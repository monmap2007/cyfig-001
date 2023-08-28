import numpy as np
import cv2
#lấy camera từ webcam
video = cv2.VideoCapture("Man1.mp4")
#trích xuất thư viện có dữ liệu sẵn
eye_detect = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#Hàm chạy liên tục để mở cửa sổ camera
while True:
    #video đọc đc gán cho frame
    _, frame = video.read()
    #cho thư viện tracking đc gán vào faces
    faces = face_detect.detectMultiScale(frame,1.3,5)
    #Truy xuất tọa độ của vật thể
    for (x,y,w,h) in faces:
        #thấy vật thể vẽ ô vuông quanh phần tracked
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        roi_frame = frame[y: y+h , x:x + w]
        eyes = eye_detect.detectMultiScale(roi_frame,1.08,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #show video đã đc track
    cv2.imshow("frame",frame)
    #nút tắt cửa sổ camera
    if cv2.waitKey(1) == ord("q"):
        break
#giải phóng video
video.release()
#làm sạch bộ nhớ
cv2.destroyAllWindows()