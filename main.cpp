#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>//neural nets
#include <stdlib.h>

////1.create a cascade classifier obj, this is the rules for how the obj are intdentified
//cv::CascadeClassifier faceCascade;//create a cascadeclassifier obj
////CNN 1. create classification vectors for CNN
//std::vector<std::string> genderList = { "Male", "Female" };
//std::vector<std::string> ageList = { "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)" };
//
//void detectFaceAndDisplay(cv::Mat& p_pFrame) {
//	//10.1 make copy of the frame and make it gray plus normalise the brightness and increase the contrast
//	cv::Mat frameGray;
//	cv::cvtColor(p_pFrame, frameGray, cv::COLOR_BGR2GRAY);//create new var that gets a grayed out version of the frame
//	cv::equalizeHist(frameGray, frameGray); //normalizes the brightness and increases the contrast of the img
//
//	//10.2 create a vector that can store rectangles and call the function that detects the faces and saves them into the vector as rectangles
//	std::vector<cv::Rect> faces;
//	faceCascade.detectMultiScale(frameGray, faces); // detects the face in the frame and returns them as a vector of rectangle into faces vector above
//
//	//10.3 a for loop that creates the low left and high right points of the face and draws a rectangle into the origianl frame
//	for (int i = 0; i < faces.size(); i++) { // for all detected faces in frame that have been stored in vector faces
//		cv::Point pt1(faces[i].x, faces[i].y);//creates point variable of the center of the face
//		cv::Point pt2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
//
//		cv::rectangle(p_pFrame, pt1, pt2, cv::Scalar(255, 0, 255), 2);//draws box around face
//
//		//CNN 3.get RIO
//		//cv::Mat frameGrayRIO = p_pFrame(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
//
//		
//	}
//
//
//
//
//	//10.4 display the frame into the window
//	cv::imshow("CameraWin", p_pFrame);
//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
//
//}

int main(int, char**) {
	//std::vector <int> vecTest;
	//for (int i = 0; i < 100000; ++i) {
	//	vecTest.push_back(i);
	//}

	//for (auto it : vecTest) {
	//	std::cout << it << std::endl;
	//}

	//std::cout << vecTest.data();


	//FACEDETECT 1. get path and create net
	std::string faceDetectProtoPath("C:\\Users\\nstrothoff\\Desktop\\openCV\\CNN\\face_detect.prototxt");
	std::string faceDetectCoffePath("C:\\Users\\nstrothoff\\Desktop\\openCV\\CNN\\face_detect.caffemodel");
	cv::dnn::Net faceDetectNet = cv::dnn::readNetFromCaffe(faceDetectProtoPath, faceDetectCoffePath);

	//IMAGE 1. new mat with images and get heigh width
	cv::Mat male = cv::imread("C:\\Users\\nstrothoff\\Desktop\\openCV\\img\\testm.jpg", -1);
	cv::Mat female = cv::imread("C:\\Users\\nstrothoff\\Desktop\\openCV\\img\\testf.jpg", -1);
	cv::Mat twoM = cv::imread("C:\\Users\\nstrothoff\\Desktop\\openCV\\img\\twoM.jpg", -1);
	//FACEDETECT 2.get dimensions of the images
	int mHeight = male.rows;
	int mWidth = male.cols;
	int fHeight = female.rows;
	int fWidth = female.cols;
	int TMHeight = twoM.rows;
	int TMWidth = twoM.cols;

	//FACEDETECT 3. create blobs(preprocess) from images
	cv::Mat mBlob = cv::dnn::blobFromImage(male, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
	cv::Mat fBlob = cv::dnn::blobFromImage(female, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
	cv::Mat TMBlob = cv::dnn::blobFromImage(twoM, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));

	//FACEDETECT 4. give the blob to the CNN and run the CNN, save info 200x7 array in Mat
	faceDetectNet.setInput(mBlob);
	cv::Mat faceDetectResult;
	faceDetectNet.forward(faceDetectResult);
	//FACEDETECT 5. Extract the content form the first Mat and read it into a second so its easily accessable
	cv::Mat facesData(faceDetectResult.size[2], faceDetectResult.size[3], CV_32F, faceDetectResult.ptr<float>());

	/*std::cout << "     img_id     " << "is_face     " << "confidence     " << "left     " << "top     " << "right     " << "bottom     " << std::endl;
	for (int i = 0; i < facesData.rows; i++) {
		std::cout << i << "     ";
		for (int j = 0; j < facesData.cols; j++) {
			std::cout << facesData.at<float>(i,j) << "     ";
		}
		std::cout << std::endl;
	}*/
	
	float confidence;
	//FACEDETECT 6. create vector of int vector to store the rectangles around the faces
	std::vector<std::vector<int>> faceRects;
	//FACEDETECT 7. for every face get create a rectangle around it and store the position of the rectangle
	for (int i = 0; i < facesData.rows; i++) {
		//get and check the probability that there is a face
		confidence = facesData.at<float>(i, 2);
		if (confidence < 0.9) { continue; }

		//get the position left/right/top/bottom of rect in downscaled img and multiply it by the real img size
		int x1 = static_cast<int>(facesData.at<float>(i, 3) * TMWidth);//left
		int y1 = static_cast<int>(facesData.at<float>(i, 4) * TMHeight);//top
		int x2 = static_cast<int>(facesData.at<float>(i, 5) * TMWidth);//right
		int y2 = static_cast<int>(facesData.at<float>(i,6) * TMHeight);//bottom
		
		//save rectangle into vector and draw into onto original img
		std::vector<int> box = { x1,y1,x2,y2 };
		faceRects.push_back(box);
		cv::rectangle(twoM, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
	}

	cv::imshow("pic", twoM);

	cv::waitKey(0);




























































	//blobFromImage(): image need preprocessing before classification which are mean subtraction and scaling by factor
	//mean subtraction needed to handle illunination(lighting) changes on the subject
	//scaling is normalizing the subtraction 
	//blobFromImage(img, scalingfactor, size, mean, swapRB);
	//img = input, scalingfactor = normalization, size = spatial size that CNN expects (224x224, 227x227, 229x229
	//mean = subtraction value either single value or 3-tuple supplied as (R,G,B)
	//swapRB = swaps from BGR openCV standard to RGB for CNN
	//blobFromImages(); for multiple images and video






	////CNN 2. get the path to the prototxt and caffemodel files
	//std::string genderProtoPath("C:\\Users\\nstrothoff\\Desktop\\openCV\\CNN\\gender_deploy.prototxt");
	//std::string genderCoffePath("C:\\Users\\nstrothoff\\Desktop\\openCV\\CNN\\gender_net.caffemodel");
	//cv::dnn::Net genderNet = cv::dnn::readNetFromCaffe(genderProtoPath, genderCoffePath);
	//if (genderNet.empty()) {
	//	std::cout << "The neural net coun't be loaded";
	//	return -1;
	//}

	//

	//if (male.empty() || female.empty()) {
	//	std::cout << "One of the images didn't load correctly";
	//	return -1;
	//}
	//

	//cv::namedWindow("male", cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("female", cv::WINDOW_AUTOSIZE);

	//cv::imshow("male", male);
	//
	//genderNet.setInput(male);
	//std::vector<float> genderProb = genderNet.forward();
	//auto genderResultIndex1 = std::distance(genderProb.begin(), std::max_element(genderProb.begin(), genderProb.end()));
	//std::cout << "This is a " << genderList[genderResultIndex1] << std::endl;

	//std::this_thread::sleep_for(std::chrono::seconds(10));

	//cv::imshow("female", female);

	//genderNet.setInput(female);
	//genderProb = genderNet.forward();
	//auto genderResultIndex2 = std::distance(genderProb.begin(), std::max_element(genderProb.begin(), genderProb.end()));
	//std::cout << "This is a " << genderList[genderResultIndex2] << std::endl;

	//cv::waitKey(0);
	////////////////////////////////////////////////////////////////////////////////////////////////
	////2. get path to the xml file with the trained classifiered 
	//std::string faceCascadePath("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
	////CNN 2. get the path to the prototxt and caffemodel files
	//std::string genderProtoPath("C:\\Users\\nstrothoff\\Desktop\\openCV\\CNN\\gender_deploy.prototxt");
	//std::string genderCoffePath("C:\\Users\\nstrothoff\\Desktop\\openCV\\CNN\\gender_net.caffemodel");
	//cv::dnn::Net genderNet = cv::dnn::readNet(genderCoffePath, genderProtoPath);
	////3.load the file with classifiers into the cascade classifier obj if doesnt work exit
	//if (!faceCascade.load(faceCascadePath)){
	//	std::cout << "Error loading face cascade!";
	//	return -1;
	//}

	////4.create obj for each frame image and create window where you'll see yourself
	//cv::namedWindow("CameraWin", cv::WINDOW_AUTOSIZE); 

	////5.create obj that lets your capture video or camera footage 1 for second camera
	//cv::VideoCapture cap(0);

	////6. check if the camera is opened and the program can recieve data from it working if not exit
	//if (!cap.isOpened()) {
	//	std::cout << "Couldn't open camera!!!";
	//	return -1;
	//}
	//std::cout << "Camera open!!" << std::endl;

	////7. Enter infinite loop that caputers a new 60 fps
	//cv::Mat cFrame;
	//int genderResultIndex;
	//while (true) {
	//	//8. read frame into obj
	//	cap.read(cFrame);

	//	//9.check if frame is empty if yes break
	//	if ( cFrame.empty()) {
	//		std::cout << "ERR frame empty";
	//		cap.release();
	//		break;
	//	}
	//	
	//	//10. call for the face detection and display of in window
	//	detectFaceAndDisplay(cFrame);

	//	//CNN
	//	genderNet.setInput(cFrame);
	//	std::vector<float> genderProb = genderNet.forward();
	//	genderResultIndex = std::distance(genderProb.begin(), std::max_element(genderProb.begin(), genderProb.end()));
	//	std::cout << genderList[genderResultIndex];

	//	//11. wait for the esc key to be pushed to close the window and break the loop
	//	if (cv::waitKey(5) == 27) {
	//		break;
	//	}
	//}

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

























	/*
	cv::Mat frame; //obj where you will add camera frames into
	cv::namedWindow("CameraWin", cv::WINDOW_AUTOSIZE); //create window for the camera frames to be displayed into
	//Capture video from camera
	//VideoCapture name();
	//Params: are device index for camera or path to video file
	//device index if one camera connected the 0 is the first one 1 second camera
	cv::VideoCapture cap(1);
	//.isOpened method checks if the camera has been opened
	if (!cap.isOpened()) {
		std::cout << "Couldn't open camera!!!";
		return -1;
	}
	if (cap.isOpened()) {
		std::cout << "Camera open!!";
	}

	//Create a endless for loop
	while(true) {
		//read a new frame from camera and store it into frame
		cap.read(frame);
		//check if frame exists
		if (frame.empty()) {
			std::cout << "ERR frame empty";
			cap.release();
			break;
		}
		//show camera frames
		cv::imshow("CameraWin", frame);
		if (cv::waitKey(5) == 27) {
			break;
		}
	}

	//waitKey(delay)
	//function listens to the keyboard the whole time if the delay is still active it will return the key pressed if not it will return -1 when key is pressed
	

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//std::string foo = "asdf"; if you create a string like this it first creates a string at 'string foo' and then a second string at '= "asdf"'
	// above method wastes time and storage when being created 
	//std::string foo("asdf"); if you create a string this way then you will only create one string that is filled with "asdf" more efficient
	
	//Read an image into program
	//imread("filepath", flag); flag for how to represent img
	cv::Mat imgColor = cv::imread("C:\\Users\\nstrothoff\\Desktop\\openCVProject\\img\\test.jpg", 1);//1 or IMREAD_COLOR
	cv::Mat imgGrayscale = cv::imread("C:\\Users\\nstrothoff\\Desktop\\openCVProject\\img\\test.jpg", 0);//0 or IMREAD_GRAYSCAPE
	cv::Mat imgUnchanged = cv::imread("C:\\Users\\nstrothoff\\Desktop\\openCVProject\\img\\test.jpg", -1);//-1 or IMREAD_UNCHANGED

	//Create windows for images to be shown
	//namedWindow("name", size);
	// size: WINDOW_AUTOSIZE
	cv::namedWindow("Color Flag Img", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Grayscale Flag Img", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Unchanged Flag Img", cv::WINDOW_AUTOSIZE);

	// Display the image.
	//imshow(windowName, image); 
	//windowName is namne of window it should be displayed into
	//imshow() is designed to be used with waitKey(), destroyAllWindows(), destroyWindow() functions
	cv::imshow("Color Flag Img", imgColor);
	cv::imshow("Grayscale Flag Img", imgGrayscale);
	cv::imshow("Unchanged Flag Img", imgUnchanged);
  
	// Wait for a keystroke. 
	// waitKey(time); time in milliseconds before is closed
	//keyboard-bindng function
	//if 0 passed it waits indefinitely
	//or pass a specifi key to wait for key trigger
	cv::waitKey();
 
	// Destroys all the windows created   
	//destroyAllWindows();
	//no params: destroyes all windows
	//window name as param: destroys that window
	cv::destroyAllWindows();
 
	// Write the image in the same directory
	// imwrite("path+name", imgVarName);
	// returns true if successful
	cv::imwrite("C:\\Users\\nstrothoff\\Desktop\\openCVProject\\img\\grayscale.jpg", imgGrayscale);*/
	return 0;
}
  