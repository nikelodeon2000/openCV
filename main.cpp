#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

//1.create a cascade classifier obj, this is the rules for how the obj are intdentified
cv::CascadeClassifier faceCascade;//create a cascadeclassifier obj

void detectFaceAndDisplay(const cv::Mat& p_pFrame) {
	//10.1 make copy of the frame and make it gray plus normalise the brightness and increase the contrast
	cv::Mat frameGray;
	cv::cvtColor(p_pFrame, frameGray, cv::COLOR_BGR2GRAY);//create new var that gets a grayed out version of the frame
	cv::equalizeHist(frameGray, frameGray); //normalizes the brightness and increases the contrast of the img

	//10.2 create a vector that can store rectangles and call the function that detects the faces and saves them into the vector as rectangles
	std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(frameGray, faces); // detects the face in the frame and returns them as a vector of rectangle into faces vector above

	//10.3 a for loop that creates the low left and high right points of the face and draws a rectangle into the origianl frame
	for (int i = 0; i < faces.size(); i++) { // for all detected faces in frame that have been stored in vector faces
		cv::Point pt1(faces[i].x, faces[i].y);//creates point variable of the center of the face
		cv::Point pt2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

		cv::rectangle(p_pFrame, pt1, pt2, cv::Scalar(255, 0, 255), 2);//draws box around face
	}

	//10.4 display the frame into the window
	cv::imshow("CameraWin", p_pFrame);
	std::this_thread::sleep_for(std::chrono::milliseconds(1));

}

int main(int, char**) {
	//2. get path to the xml file with the trained classifiered 
	std::string faceCascadePath("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
	
	//3.load the file with classifiers into the cascade classifier obj if doesnt work exit
	if (!faceCascade.load(faceCascadePath)){
		std::cout << "Error loading face cascade!";
		return -1;
	}

	//4.create obj for each frame image and create window where you'll see yourself
	cv::namedWindow("CameraWin", cv::WINDOW_AUTOSIZE); 

	//5.create obj that lets your capture video or camera footage 1 for second camera
	cv::VideoCapture cap(0);

	//6. check if the camera is opened and the program can recieve data from it working if not exit
	if (!cap.isOpened()) {
		std::cout << "Couldn't open camera!!!";
		return -1;
	}
	std::cout << "Camera open!!" << std::endl;

	//7. Enter infinite loop that caputers a new 60 fps
	cv::Mat cFrame;
	while (true) {
		//8. read frame into obj
		cap.read(cFrame);

		//9.check if frame is empty if yes break
		if ( cFrame.empty()) {
			std::cout << "ERR frame empty";
			cap.release();
			break;
		}
		
		//10. call for the face detection and display of in window
		detectFaceAndDisplay(cFrame);

		//11. wait for the esc key to be pushed to close the window and break the loop
		if (cv::waitKey(5) == 27) {
			break;
		}
	}






























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
  