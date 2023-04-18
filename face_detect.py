import cv2


img = cv2.imread("PRO-C106-Reference-Code-main/boy.jpg")  

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("PRO-C106-Reference-Code-main/haarcascade_frontalface_default.xml")

faces = face_cascade.detectMultiScale(gray)

print(faces) #
print(img)


cv2.imshow("image", img)
