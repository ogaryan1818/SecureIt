import threading
import cv2
from deepface import DeepFace
import os

directory = r"C:\Users\rocki\OneDrive - vitbhopal.ac.in\Desktop\PROJECTS\Python\Face2"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

face_match = False


def checkface(frame):
    global face_match

    for filename in os.listdir(directory):
        absolutePath = os.path.join(os.getcwd(), 'Ref', 'reference.jpg');
        reference = cv2.imread(absolutePath)
        print(reference)
        try:
            if DeepFace.verify(frame, reference.copy())["verified"]:
                face_match = True
                return
            else:
                face_match = False
        except ValueError:
            face_match = False


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=checkface, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(
                frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
            )
        else:
            cv2.putText(
                frame,
                "No Match!",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
            )
        cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
