import numpy as np
import cv2

# Initialize camera
cap = cv2.VideoCapture(0)

# Load the haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('/home/anurag/Downloads/haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = '/media/anurag/Anurag/data_set/'

file_name = input("Enter the name of the person: ")


while True:
                
	ret, frame = cap.read()
	if ret == False:
		continue
	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	k=1

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	

	faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

	# update the frame number
	skip += 1

	for face in faces[:1]:
		x, y, w, h = face

		# Get the face ROI
		offset = 7
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		
		if skip % 10 == 0:
			face_data.append(face_section)
			print(len(face_data))

		# Display the face ROI
		cv2.imshow(str(k), face_section)
		k += 1

		# Draw rectangle in the original image
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Convert face list to numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

# Save the dataset in filesystem..........syntax is np.save('string which represents file path concatenated with file_name',ndarray)
np.save(dataset_path + file_name, face_data) #further np.load('string') will be used for loading the ndarray
print( "Dataset saved at: {}".format(dataset_path + file_name + '.npy'))

cv2.destroyAllWindows()

