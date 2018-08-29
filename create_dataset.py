import imutils
import cv2
from imutils.video import VideoStream
import argparse
import os
import math
ap = argparse.ArgumentParser()


vs = VideoStream(src=0).start()
writer = None
#time.sleep(2.0)
ap.add_argument("-o", "--dataset", type=str,
	help="path to output frames")
args = vars(ap.parse_args())
directry = 'custom_dataset/'+args['dataset']
print (directry)
count = 0
ranges =  int(math.pow(10,5))
while count<=500:
    for i in range(ranges):
        frame = vs.read()
    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame", frame)
    if not os.path.exists(directry):
        os.makedirs(directry)
    cv2.imwrite(directry+"/frame%d.jpg" % count, frame)
    rgb = imutils.resize(frame, width=200)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count +=1
    
    

cv2.destroyAllWindows()
vs.stop()
    
