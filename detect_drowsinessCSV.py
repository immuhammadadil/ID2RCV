# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
#Author muhammad Adil
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from time import sleep, strftime, time


import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import time
import os



def updateEAR(drowsy_status_val, distracted_status_val, ear_val, total_val, xcord, ycord):
	if drowsy_status_val==0:
		drowsy_status=1
	elif drowsy_status_val==1:
		drowsy_status=0
	if distracted_status_val==0:
		distracted_status=1
	elif distracted_status_val==1:
		distracted_status=0
	log.write("{0},{1},{2},{3},{4},{5},{6}\n".format(strftime("%Y-%m-%d %H:%M:%S"),str(drowsy_status),str(distracted_status), int(total_val), float(ear_val), float(xcord), float(ycord)))

	
	
	
	
	cv2.putText(screen, "Drowsy Status: {}".format(drowsy_status), (10, 280),
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
	cv2.putText(screen, "Distraction Status: {}".format(distracted_status), (10, 310),
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
	
		#log.write("{0},{1}\n".format(strftime("%Y-%m-%d %H:%M:%S"),str(distracted_status)))

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)
	#playsound('/home/adil/NUST/DrowsinessDetecion/drowsiness-detection/alarm.wav')

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default='/home/adil/NUST/DrowsinessDetecion/drowsiness-detection/alarm.wav',
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_THRESH_BLINK = 0.2
EYE_AR_CONSEC_FRAMES = 48
Distraction_thresh=45
Distraction_counter=0
EYE_AR_CONSEC_FRAMES_BLINK = 3
TOTAL=0
alert_level=5
drowsy_status_val=0
distracted_status_val=0






# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
COUNTER_BLINK=0
ALARM_ON = False


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



# start the video stream thread
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture('Subject1.mp4')
#vs = VideoStream(src=args["webcam"]).start()
#time.sleep(1.0)


#------------Pose Estimation pre calculations-------------------------------------------------------
# 400x225 to 1024x576
frame_width = 1024
frame_height = 576

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (359, 391),     # Nose tip 34
                            (399, 561),     # Chin 9
                            (337, 297),     # Left eye left corner 37
                            (513, 301),     # Right eye right corne 46
                            (345, 465),     # Left Mouth corner 49
                            (453, 469)      # Right mouth corner 55
                        ], dtype="double")

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip 34
                            (0.0, -330.0, -65.0),        # Chin 9
                            (-225.0, 170.0, -135.0),     # Left eye left corner 37
                            (225.0, 170.0, -135.0),      # Right eye right corne 46
                            (-150.0, -150.0, -125.0),    # Left Mouth corner 49
                            (150.0, -150.0, -125.0)      # Right mouth corner 55

                        ])



#------------End Pose Estimation pre calculations------------------------------------------

#creating CSV file to record data

with open ("/home/adil/NUST/CSVversion/Status_Log.csv", "a") as log:

# loop over frames from the video stream
	while(vs.isOpened()):
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
		ret, frame = vs.read()
	
		frame = imutils.resize(frame, width=450)
		frame1 = imutils.resize(frame, width=450)
		screen = imutils.resize(frame, width=600)
		frame2 = imutils.resize(frame, width=450) 
		
	#frame2 = imutils.resize(frame, width=1024, height=576)
	
	
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		size = gray.shape
	# detect faces in the grayscale frame
		rects = detector(gray, 0)

		if len(rects) > 0:
			text = "{} face(s) found".format(len(rects))
			cv2.putText(frame1, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 0, 255), 2)


		cv2.putText(screen, "Real Time Parameters!", (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

		cv2.putText(screen, "Threshold for Sleep Detection= 48 Frames", (10, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)



	# loop over the face detections
		for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

		


		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
			

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			
		#Mark the facial landmarks
			for (i, (x, y)) in enumerate(shape):
				cv2.circle(frame1, (x, y), 1, (0, 0, 255), -1)
				cv2.putText(frame1, str(i + 1), (x - 10, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
			

		# loop over the face parts individually
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
				clone = frame.copy()
				cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)
 
		# loop over the subset of facial landmarks, drawing the
		# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
#--------------------------------------------------------------------------------------------------
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH_BLINK:
				COUNTER_BLINK += 1
 
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
			else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
				if COUNTER_BLINK >= EYE_AR_CONSEC_FRAMES_BLINK:
					TOTAL += 1
 
			# reset the eye frame counter
				COUNTER_BLINK = 0

		
#--------------------------------------------------------------------------------------------------

			for (i, (x, y)) in enumerate(shape):
				cv2.circle(frame1, (x, y), 1, (0, 0, 255), -1)
				cv2.putText(frame1, str(i + 1), (x - 10, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
#=====================================================================================================
				if i == 33:
                                        #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        	image_points[0] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        	cv2.circle(frame2, (x, y), 1, (0, 255, 0), -1)
                                        	cv2.putText(frame2, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
				elif i == 8:
                    #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        	image_points[1] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        	cv2.circle(frame2, (x, y), 1, (0, 255, 0), -1)
                                        	cv2.putText(frame2, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
				elif i == 36:
                                        #something to our key landmarks
                                        # save to our new key point list
                                        # i.e. keypoints = [(i,(x,y))]
                                        	image_points[2] = np.array([x,y],dtype='double')
                                        # write on frame in Green
                                        	cv2.circle(frame2, (x, y), 1, (0, 255, 0), -1)
                                        	cv2.putText(frame2, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
				elif i == 45:
                                        #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        	image_points[3] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        	cv2.circle(frame2, (x, y), 1, (0, 255, 0), -1)
                                        	cv2.putText(frame2, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
				elif i == 48:
                    #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        	image_points[4] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        	cv2.circle(frame2, (x, y), 1, (0, 255, 0), -1)
                                        	cv2.putText(frame2, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
				elif i == 54:
                    #something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                                        	image_points[5] = np.array([x,y],dtype='double')
                    # write on frame in Green
                                        	cv2.circle(frame2, (x, y), 1, (0, 255, 0), -1)
                                        	cv2.putText(frame2, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
				else:
                    #everything to all other landmarks
                    # write on frame in Red
                                        	cv2.circle(frame2, (x, y), 1, (0, 0, 255), -1)
                                        	cv2.putText(frame2, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
			focal_length = size[1]
			center = (size[1]/2, size[0]/2)
			camera_matrix = np.array([[focal_length,0,center[0]],[0, focal_length, center[1]],[0,0,1]], dtype="double")

                        #print "Camera Matrix :\n {0}".format(camera_matrix)

			dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
			(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)#flags=cv2.CV_ITERATIVE)

                        #print "Rotation Vector:\n {0}".format(rotation_vector)
                        #print "Translation Vector:\n {0}".format(translation_vector)
                        # Project a 3D point (0, 0 , 1000.0) onto the image plane
                        # We use this to draw a line sticking out of the nose_end_point2D
			(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),rotation_vector, translation_vector, camera_matrix, dist_coeffs)
			for p in image_points:
				cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

			p1 = ( int(image_points[0][0]), int(image_points[0][1]))
			p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		#Print p2 value
			p2_str=str(p2)
			cv2.putText(screen, "Face Direction "+p2_str, (10,150),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
			cv2.line(frame2, p1, p2, (255,0,0), 2)



		#Extracting Y & Y coordinates of nose=============
			xcord= int(nose_end_point2D[0][0][0])
			ycord= int(nose_end_point2D[0][0][1])

			xcord_str=str(xcord)
			ycord_str=str(ycord)

		#cv2.putText(screen, "xcord "+xcord_str, (10,250),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		#cv2.putText(screen, "ycord "+ycord_str, (10,280),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

		#End Extracting x & Y coordinates===============
		

		#Distarction Detection==========================

			if(xcord<100)|(xcord>360):
				Distraction_counter += 1
				if Distraction_counter >= Distraction_thresh:
					cv2.putText(screen, "Driver Distracted!", (10, 250),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					distracted_status_val=1
					os.system('spd-say "Distraction Detected"')

			


			else: 
				distracted_status_val=0
				Distraction_counter=0
			


#====================================================================================================			
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
				time0=time.clock()
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
					if not ALARM_ON:
						ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
						if args["alarm"] != "":
							t = Thread(target=sound_alarm,
								args=(args["alarm"],))
							t.deamon = True
							t.start()
							drowsy_status_val=1
							os.system('spd-say "Drowsiness Detected"')					
						
					a=int (COUNTER)
					b=str (a)
				
				
					ctime=time.clock()-time0
					c=str(ctime-time0)
								
				# draw an alarm on the frame
					cv2.putText(screen, "DROWSINESS ALERT!", (10, 180),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
					#cv2.putText(screen, "Eyes closed for "+b+" Frames", (10, 210),
					#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
				
				
		


		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
			else:
				COUNTER = 0
				ALARM_ON = False
				time0=0
				c=0
				alert_level=5
				drowsy_status_val=0
		
		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
			cv2.putText(screen, "Eye Aspect Ratio: {:.2f}".format(ear), (10, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
			#log.write("{0},{1}\n".format(strftime("%Y-%m-%d %H:%M:%S"),str(ear)))

			cv2.putText(screen, "Blinks: {}".format(TOTAL), (10, 120),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
 		
		
		#str_alert=str(alert_level)		
		#cv2.putText(screen, "Alertness Level "+str_alert, (10, 280),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	
			updateEAR(drowsy_status_val, distracted_status_val, ear, TOTAL, xcord, ycord)
			


	# show the frame
		cv2.imshow("Frame", frame)
		cv2.imshow("Frame1", frame1)
		cv2.imshow("screen", screen)
		cv2.imshow("Frame2", frame2)			
		key = cv2.waitKey(1) & 0xFF
	

 
	# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	#window = tkinter.Tk()
	#window.mainloop()
# do a bit of cleanup
cv2.destroyAllWindows()
