import cv2

# set parameters
frame_size = (400, 250)
min_area = 1000

# Load the video
cap = cv2.VideoCapture("Files/sample.mp4")

avg = None

# Star the loop for video
while True:
	# capture the frame and the return status from video
	ret, frame = cap.read()

	# if return is true then continue otherwise break the loop
	if ret:
		# resize the frame
		frame = cv2.resize(frame, dsize=frame_size)

		# convert to gray
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# blur the image
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		# if the average frame is None, initialize it
		if avg is None:
			avg = gray.copy().astype("float")
			continue

		# accumulate the weighted average between the current frame and
		# previous frames, then compute the difference between the current
		# frame and running average
		cv2.accumulateWeighted(gray, avg, 0.5)
		frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

		# threshold the delta image, dilate the thresholded image to fill
		# in holes, then find contours on thresholded image
		thresh = cv2.threshold(frameDelta, 5, 255,
			cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)

		contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[1]

		# loop over the contours
		for contour in contours:
			# if the contour is too small, ignore it
			if cv2.contourArea(contour) < min_area:
				continue

			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(contour)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# display the frame
		cv2.imshow("Motion Dectection", frame)

		# if the `q` key is pressed, break from the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()