import numpy as np
import cv2
import os
########## KNN CODE ############
def distance(v1, v2):
	# Eucledian distance
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################

# Initialize camera
cap = cv2.VideoCapture(0)

# Load the haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('/home/anurag/Downloads/haarcascade_frontalface_alt.xml')

dataset_path ='/media/anurag/Anurag/data_set/'

face_data = []
labels = []
class_id = 0
names={}

# Dataset prepration
for fx in os.listdir(dataset_path):      #will return list of strings
      if fx.endswith('.npy'):        # property of a string as fx is string
                data_item = np.load(dataset_path + fx)
                #print(data_item,type(data_item),)  #prints matrix and class<'numpy.nd array'>
                face_data.append(data_item)
                names[class_id]=fx[:-4]
                target = class_id * np.ones((data_item.shape[0],))  #assigning labels as we had not assigned while saving dataset
                class_id += 1   # when next file will be loaded,label assigned will be changed so incremented
                labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)  # Here we are combining all images ie....(x1 images of label 0)+(x2 images of label 1)..........=(x1+x2+.....xn)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))# Here we have reshaped for converting 1D matrix into 2D matrix
#print(face_labels.shape)
#print (face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
#print('Shape of trainset is',trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	ret, frame = cap.read()
	if ret == False:
		continue
	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for face in faces:
		x, y, w, h = face

		# Get the face ROI
		offset = 7
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))
		#print('shape of face_section is',face_section.shape)
		#print('shape of face_section after flattening is',face_section.flatten().shape)

		out = knn(trainset, face_section.flatten()) # as testing data was 1D matrix in knn.......hence face_section also should be 1D matrix

		# Draw rectangle in the original image and put text in it 
		cv2.putText(frame, names[int(out)],(x,y-10), font, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()