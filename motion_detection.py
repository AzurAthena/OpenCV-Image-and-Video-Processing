import cv2

# set parameters
frame_size = (400, 250)
min_area = 1000
max_area = 5000
threshold = 15
delta_iterations = 3
update_interval = 100

# Load the video
cap = cv2.VideoCapture("Files/sample.mp4")

initial = None

# Star the loop for video
index = 0
while cap.isOpened():
	# Keep track of frames with index
	index += 1

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

		# intialise the initial frame
		if initial is None:
			initial = gray.copy()
			continue

		# compute the difference between the current frame and the average
		frameDelta = cv2.absdiff(gray, initial)

		# threshold the delta image
		thresh = cv2.threshold(frameDelta, threshold, 255,
			cv2.THRESH_BINARY)[1]

		# dilate the thresholded image to fill in holes
		thresh = cv2.dilate(thresh, None, iterations=delta_iterations)

		# find contours on thresholded image
		contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[1]

		# loop over the contours
		for contour in contours:
			# if the contour is too small, ignore it
			if (cv2.contourArea(contour) < min_area) or (cv2.contourArea(contour) > max_area):
				continue

			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(contour)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# update the average after few frames
		if index % update_interval == 0:
			print('Updating after frames', index)
			initial = gray.copy()

		# display the frame
		cv2.imshow("Motion Dectection", frame)

		# if the `q` key is pressed, break from the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()