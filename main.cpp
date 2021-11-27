#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//Webcam

int main() {
    //Create webcam capture and apply correct settings
    VideoCapture cap(2, CAP_V4L);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G')); // Opens with Mjpg format to allow highest resolution
    cap.set(CAP_PROP_FRAME_WIDTH, 3200);
    cap.set(CAP_PROP_FRAME_HEIGHT, 856); // sets resolution to 3200 x 856
    cap.set(CAP_PROP_FPS, 60); // sets framerate to 60 fps (may not be nessessary)

    // Declare images as matricies
    Mat frame;

    Mat rightCam;
    Mat leftCam;
    Mat frontRightCam;
    Mat frontLeftCam;
    Mat upCam;

    Mat rightCam_controller;
    Mat leftCam_controller;
    Mat frontRightCam_controller;
    Mat frontLeftCam_controller;
    Mat upCam_controller;

    // Testing Variables (to delete)
    Mat stereoImg;
    Mat susImg;

    while (true) {
      cap.read(frame); //setting frame to the current frame of video

      /* Note: the camera sends both a black and white and an IR image in alternating frames.          */
      /* Each is functionally 30 fps                                                                   */
      /* There are debug pixels on the top that change depending on which view it's in, IR or visible  */
      if ((int)frame.at<Vec3b>(1,1)[0] == 255) {  // check if it's on the infared or normal view by looking at the debug pixels

        // Crop the single image to each of the 5 cameras
        rightCam = frame(Rect(2560,375,640,480));
        leftCam = frame(Rect(640,375,640,480));
        frontRightCam = frame(Rect(1280,375,640,480));
        frontLeftCam = frame(Rect(1920,375,640,480));
        upCam = frame(Rect(0,375,640,480));

        // Rotate images to correct orientation
        rotate(leftCam, leftCam, ROTATE_90_COUNTERCLOCKWISE);
        rotate(rightCam, rightCam, ROTATE_90_COUNTERCLOCKWISE);
        rotate(frontLeftCam, frontLeftCam, ROTATE_90_COUNTERCLOCKWISE);
        rotate(frontRightCam, frontRightCam, ROTATE_90_COUNTERCLOCKWISE);

        Ptr<ORB> keyPointDetector = ORB::create(); // Create an orb key point detector
        /* IMPORTANT This is where you change what a keypoint is          */
        /* The paramaters of the create() function can change how         */
        /* the detector detects keypoints and what it considers keypoints */

        vector<KeyPoint> keypoints_left, keypoints_right; // Keypoints for front left and front right cameras

        // Detects keypoints
        keyPointDetector->detect(frontLeftCam, keypoints_left);
        keyPointDetector->detect(frontRightCam, keypoints_right);

        // Create descriptors. These describe keypoints and can be used to see if two keypoints match
        Mat descriptors_left, descriptors_right;


        Ptr<DescriptorExtractor> extractor = ORB::create(); // Creates an orb to extract descriptors from keypoints
        /* create() here can also have other parameters */

        // Extracts descriptors from keypoints
        extractor->compute(frontLeftCam, keypoints_left, descriptors_left);
        extractor->compute(frontRightCam, keypoints_right, descriptors_right);

        // Convert descriptors to CV_32f format so they can be used for Flann
        descriptors_left.convertTo(descriptors_left, CV_32F);
        descriptors_right.convertTo(descriptors_right, CV_32F);

        // Finds matching keypoints
        FlannBasedMatcher matcher;
        vector<DMatch> matches;
        matcher.match( descriptors_left, descriptors_right, matches);


        // Remove matches that are not horizontaly across from each other
        // Because of the stario cameras, matches should only be horizontally accross from each other
        int verticalDifference = 5; // can be changed for sensitivity (default 5)
        std::vector<DMatch> gooder_matches;

        for(DMatch i : matches) {
          if(abs(keypoints_left[i.queryIdx].pt.y-keypoints_right[i.trainIdx].pt.y) <= verticalDifference) {
            gooder_matches.push_back(i);
          }
        }

        // Draws debug lines to stario image camera
        drawMatches(frontLeftCam, keypoints_left, frontRightCam, keypoints_right, gooder_matches, stereoImg, Scalar(0,255,0),Scalar(0,0,255),vector<char>());

        // Shows the visble light images
        imshow("Stereo Matches", stereoImg);

        imshow("Right Camera", rightCam);
        imshow("Left Camera", leftCam);
        imshow("Front Right Camera", frontRightCam);
        imshow("Front Left Camera", frontLeftCam);
        imshow("Up Camera", upCam);

      } else { // Detects if it is showing an IR camera

        // Crop the image to the 5 cameras
        rightCam_controller = frame(Rect(2560,40,640,480));
        leftCam_controller = frame(Rect(640,40,640,480));
        frontRightCam_controller = frame(Rect(1280,40,640,480));
        frontLeftCam_controller = frame(Rect(1920,40,640,480));
        upCam_controller = frame(Rect(0,40,640,480));

        // Shows IR images
        imshow("Right Camera Controller View", rightCam_controller);
        imshow("Left Camera Controller View", leftCam_controller);
        imshow("Front Right Camera Controller View", frontRightCam_controller);
        imshow("Front Left Camera Controller View", frontLeftCam_controller);
        imshow("Up Camera Controller View", upCam_controller);
      }
      waitKey(1); // Nessessary to make sure the windows aren't instently destroyed
    }
      return 0; // Never gets called because of the while true
}
