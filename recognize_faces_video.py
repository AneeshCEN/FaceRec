# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
from scipy.spatial import distance
import face_recognition
import argparse
import imutils
from imutils import face_utils
import pickle
import time
import cv2
import dlib

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
thresh = 0.25
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# loop over frames from the video file stream
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
while True:
    frame = vs.read()
	
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=200)
    r = frame.shape[1] / float(rgb.shape[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    subjects = detect(gray, 0)
#    print ('sub lenght', len(subjects))
#    for (i, subject) in enumerate(subjects):
#        shape = predict(gray, subject)
#        shape = face_utils.shape_to_np(shape)
#        leftEye = shape[lStart:lEnd]
#        rightEye = shape[rStart:rEnd]
#        leftEAR = eye_aspect_ratio(leftEye)
#        rightEAR = eye_aspect_ratio(rightEye)
#        ear = (leftEAR + rightEAR) / 2.0
#        print ('ear is', ear)
#        leftEyeHull = cv2.convexHull(leftEye)
#        rightEyeHull = cv2.convexHull(rightEye)
#        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#        if ear<thresh:
#            cv2.putText(frame, "real face", (10, 30),
#					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#        
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

	# loop over the facial embeddings
    for encoding in encodings:
        #tic=time.time()
		# attempt to match each face in the input image to our known
		# encodings
        matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		
        name = "Unknown"

		# check to see if we have found a match
        if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
            name = max(counts, key=counts.get)
		
		# update the list of names
        names.append(name)
        #print('Time taken',time.time()-tic)
	# loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        
        sub_img = gray[top:bottom, left:right]
        subjects = detect(sub_img, 0)
        print ('sub lenght', len(subjects))
        for (i, subject) in enumerate(subjects):
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            print ('ear is', ear)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear<thresh:
                cv2.putText(frame, "real face", (10, 30),
        					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        writer = cv2.VideoWriter(args["output"], fourcc, 5,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces t odisk
    if writer is not None:
        writer.write(frame)

	# check to see if we are supposed to display the output frame to
	# the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()
