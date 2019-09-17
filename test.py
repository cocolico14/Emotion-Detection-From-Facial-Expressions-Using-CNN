import tensorflow as tf
import imutils
import numpy as np
import cv2
from skimage.feature import hog
from fast_webcam import WebcamVideoStream

W, H = 1080, 720
S = 2
FPS = 60
CNN_FPS = 3

FACE_CASCADE = cv2.CascadeClassifier('cascade.xml')
EMOTION_CLASSIFIER = tf.keras.models.load_model('fold_6_62.08%.hdf5')

FONT = cv2.FONT_HERSHEY_SIMPLEX
CATEGORIES = ["Surprise", "Happy", "Sad", "Angry", "Natural", "Fear", "Disgust"]

vs = WebcamVideoStream(src=0).start()
cnt = 0
tmp_faces = []
X = np.zeros((1, 48, 48, 1))

while 1:
    img = vs.read()
    img = imutils.resize(img, width=W, height=H)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ds_img = imutils.resize(gray, width=W//S, height=H//S)
    faces = FACE_CASCADE.detectMultiScale(ds_img, 1.3, 6)

    if len(faces) == 0:
        faces = tmp_faces
        if cnt%CNN_FPS==0:
            tmp_faces = []
    else:
        tmp_faces = faces

    for (x,y,w,h) in faces:
        x, y, w, h = x*S, y*S, w*S, h*S
        cv2.rectangle(img,(x-20, y-20),(x+w+20,y+h+20),(255,255,0),2)
        raw_face = gray[y+5:y+h-5, x+25:x+w-25]
        raw_face = cv2.resize(raw_face, (48, 48))
        if cnt%CNN_FPS == 0:
            X = np.array(raw_face).reshape(-1, 48, 48, 1)
            X = tf.keras.utils.normalize(X, axis=1)
        Y = EMOTION_CLASSIFIER.predict(X)[0]
        text = "Su:{0:.0f}% H:{1:.0f}% Sa:{2:.0f}% A:{3:.0f}% N:{4:.0f}% F:{5:.0f}% D:{6:.0f}%".format(
                Y[0]*100, Y[1]*100, Y[2]*100, Y[3]*100, Y[4]*100, Y[5]*100, Y[6]*100)
        cv2.putText(img, CATEGORIES[np.argmax(Y)], (x-10,y), FONT, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x+(w//2)-160,y+h+40), FONT, 0.5, (255,255,0), 1, cv2.LINE_AA)

    cv2.imshow('img',img)
    if cv2.waitKey(1000//FPS) & 0xFF == ord('q'):
        break

    if cnt%CNN_FPS == 0:
        cnt = 0
    
    cnt += 1
    
vs.stop()
cv2.destroyAllWindows()