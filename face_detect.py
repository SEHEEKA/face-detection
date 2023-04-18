import cv2


img = cv2.imread("boy.jpg")  

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = face_cascade.detectMultiScale(gray)

print(faces) #
print(img)


cv2.imshow("image", img)
