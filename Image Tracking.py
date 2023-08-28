import cv2

# tải cascade classifiers cho mặt và mũi
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Up ảnh
image = cv2.imread('Man.jpg')
#Kích thước mặc định - Chuyển đổi thành Full HD
new_width = 1920
new_height = 1080
resized_image = cv2.resize(image,(new_width, new_height))
#Chuyển màu của ảnh thành đen trắng
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Xác định khuôn mặt
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=5)

# Duyệt qua các tọa độ của khuôn mặt
for (x, y, w, h) in faces:
    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Giải phóng các khu vực được đánh dấu là mặt từ ảnh trắng đen
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = resized_image[y:y+h, x:x+w]
    
    # Xác định mắt từ vùng mặt đã đánh đấu
    eyes = eye_cascade.detectMultiScale(roi_gray,1.25,5)
    
    # Duyệt qua các vùng đc cho là mắt trong khu vực đc đánh dấu là mặt (Vì mắt ở trong mặt)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Faces and Eyes', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
