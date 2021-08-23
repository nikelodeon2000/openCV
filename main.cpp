#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>//neural nets
#include <stdlib.h>
#include <iterator>

void getFaces(cv::Mat &faceCopy, cv::dnn::Net faceDetectNet, std::vector<std::vector<int>> &faceBoxes) {
	//FACEDETECT 5.get dimensions of the images
	int mHeight = faceCopy.rows;
	int mWidth = faceCopy.cols;
	
	//FACEDETECT 6.create blob for net
	cv::Mat faceBlob;
	faceBlob = cv::dnn::blobFromImage(faceCopy, 1.0, cv::Size(300, 300), cv::Scalar(104, 117, 123), true, false);
	//FACEDETECT 7. input blob into network and active it to give you result
	faceDetectNet.setInput(faceBlob);
	cv::Mat faceResult = faceDetectNet.forward();
	//FACEDETECT 8.read result into new mat to make it iterable, iterate and save face coordinates into faceBoxes
	cv::Mat readableFaceResult(faceResult.size[2], faceResult.size[3], CV_32F, faceResult.ptr<float>());
	float confidence;
	for (int i = 0; i < readableFaceResult.rows; i++) {
		//get and check the probability that there is a face
		confidence = readableFaceResult.at<float>(i, 2);
		if (confidence < 0.9) { continue; }

		//get the position left/right/top/bottom of rect in downscaled img and multiply it by the real img size
		int x1 = static_cast<int>(readableFaceResult.at<float>(i, 3) * mWidth);//left
		int y1 = static_cast<int>(readableFaceResult.at<float>(i, 4) * mHeight);//top
		int x2 = static_cast<int>(readableFaceResult.at<float>(i, 5) * mWidth);//right
		int y2 = static_cast<int>(readableFaceResult.at<float>(i, 6) * mHeight);//bottom

		//save rectangle into vector and draw into onto original img
		std::vector<int> box = { x1,y1,x2,y2 };
		faceBoxes.push_back(box);
		cv::rectangle(faceCopy, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
	}
}


int main(int, char**) {

	//FACEDETECTION 1. get model paths
	std::string faceDetectProtoPath("face_detect.prototxt");
	std::string faceDetectCoffePath("face_detect.caffemodel");

	//GENDERDETECTION 1. get model paths
	std::string genderProtoPath("gender_deploy.prototxt");
	std::string genderCoffePath("gender_net.caffemodel");

	//AGEDETECTION 1. get model paths
	std::string ageProtoPath = "age_deploy.prototxt";
	std::string ageCoffePath = "age_net.caffemodel";

	//SCALR FOR GENDER AGE BLOB
	cv::Scalar MODEL_MEAN_VALUES = cv::Scalar(78.4263377603, 87.7689143744, 114.895847746);

	//GENDERDETECTION 2. create result vector for output
	std::vector<std::string> genderList = { "Male", "Female" };

	//AGEDETECTION 2. create result vector for output
	std::vector<std::string> ageList = { "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)" };

	//GENDERDETECTION 3. FACEDETECTION 2. AGEDETECTION 3. load the neural network
	cv::dnn::Net faceDetectNet = cv::dnn::readNetFromCaffe(faceDetectProtoPath, faceDetectCoffePath);
	cv::dnn::Net genderNet = cv::dnn::readNetFromCaffe(genderProtoPath, genderCoffePath);
	cv::dnn::Net ageNet = cv::dnn::readNetFromCaffe(ageProtoPath, ageCoffePath);

	//LOAD IMG
	//cv::Mat img = cv::imread("C:\\Users\\nstrothoff\\Desktop\\openCV\\img\\fwomen.jpg", 1);

	//CREATE CAMERA AND START IT UP
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cout << "Couldn't open camera!!!";
		return -1;
	}
	if (cap.isOpened()) {
		std::cout << "Camera open!!";
	}

	//FACEDETECTION 3. vector of vectors to store face boxes coordinates
	std::vector<std::vector<int>> faceBoxes;

	while (cv::waitKey(1) < 0) {
		cv::Mat img;
		//read a new frame from camera and store it into frame
		cap.read(img);
		//check if frame exists
		if (img.empty()) {
			std::cout << "ERR frame empty";
			cap.release();
			break;
		}
		//FACEDETECTION 4. create copy of img
		cv::Mat faceCopy = img.clone();
		//FACEDETECTION
		getFaces(faceCopy, faceDetectNet, faceBoxes);

		//GENDERDETECTION 4. iterate over all faces
		int i = 0;
		for (auto it : faceBoxes) {
			//GENDERDETECTION 5. cut out all faces
			cv::Rect ROI(cv::Point(faceBoxes[i][0] - 15, faceBoxes[i][1] - 15), cv::Point(faceBoxes[i][2] + 15, faceBoxes[i][3] + 15));
			cv::Mat face = img(ROI);

			//GENDERDETECTION 6. make blob of face
			cv::Mat Blob;
			Blob = cv::dnn::blobFromImage(face, 1, cv::Size(227, 227), MODEL_MEAN_VALUES, false);

			//GENDERDETECTION 7. send face blob into network
			genderNet.setInput(Blob);
			std::vector<float> genderProb = genderNet.forward();
			//GENDERDETECTION 8. find max probability of result
			int maxProbGender = std::max_element(genderProb.begin(), genderProb.end()) - genderProb.begin();;
			std::string gender = genderList[maxProbGender];

			//agedetection 4. input blob into network and get result
			ageNet.setInput(Blob);
			std::vector<float> ageProb = ageNet.forward();
			//agedetection 5. find max and save it for output
			int maxPobage = std::max_element(ageProb.begin(), ageProb.end()) - ageProb.begin();;
			std::string age = ageList[maxPobage];


			std::cout << "gender: " << gender << std::endl;

			std::string boxLabel = gender + ", " + age;
			cv::putText(faceCopy, boxLabel, cv::Point(faceBoxes[i][0], faceBoxes[i][1] - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
			i++;
		}

		cv::imshow("Frame", faceCopy);
		//CLEAR LAST FRAME FACEBOX
		faceBoxes.clear();
		if (cv::waitKey(5) == 27) {
			break;
		}
	}



	//STEPS CNN IN OPENCV
	//1. Create string vector foe the different classes Gender(male,female) Age(0-2,4-6.,8-12,15-20,25-32,38-43,48-53,60-100)
	//2. define prototxt path, weight(coffemodel) path
		//prototxt files defines the different stages of the network
	//3. read neural net with prototxt  and coffemodel
	//4. preprocess image with blob
	//4.get the Region Of Interest (face) from the image and pass it into the network setinput
	//5. perform the analysis with the network .forward and save the probabilities into a vector
	//6. find the highest probability and output the index with the highest probabaility


	//CNN WORKINGS
	//in convolutional neural networks the different hidden layers perform random convolutions and by running 
	//based on the first conolution the following hidden layer extract new features again with another convolution based on the first
	//at the end based on how many node per layer you have a huge set of images that have been derived from the origanal
	// image that all look different based on the different combinations of convolutions that have been performed
	// the images are downiszed because of the convolution kernel until they are only one pixel
	// each pixel represent a different feature that has been determined by the convolution and if its present its white if not black
	// these are the neurons at the end
	//after many iterations the weight of the kernals at the different layers are changed to determine the right outcome


}
