from imutils import face_utils
from scipy.spatial import distance
import dlib
import cv2
import vlc

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(".\shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
media = vlc.MediaPlayer("F:\capstone\Master.mp4")
media.play()
flag=0
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= detect(gray)
    length=len(faces)
    if len(faces)>0:
        media.set_pause(0)
    else:
        media.set_pause(1)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        
        shape= predict(gray,face)
        shape=face_utils.shape_to_np(shape)
        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEAR=eye_aspect_ratio(leftEye)
        rightEAR=eye_aspect_ratio(rightEye)
        ear=(leftEAR+rightEAR)/2.0
        if ear<thresh:
            flag+=1
            
                
                cv2.putText(frame,"****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                media.set_pause(1)
        else:
            flag=0
            media.set_pause(0)
    
            
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

media.stop()
cap.release()
cv2.destroyAllWindows()
    
