import cv2

stream = cv2.VideoCapture(1)

while True:
    ret, frame = stream.read()
    if ret:
        print(frame.shape)
        cv2.imshow('frame', frame)
    key = cv2.waitKey(int(1000/60))
    if key == 27:
        cv2.destroyAllWindows()
        break
        

exit()