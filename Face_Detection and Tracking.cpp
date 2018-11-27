#include "cv.hpp"
#include <iostream>

using namespace std;
using namespace cv;


int main() {

	// 1.1  Face_Detection_Forward
	/*
	CascadeClassifier face_classifier;
	VideoCapture cap("face.mp4");
	Mat frame, grayframe;
	vector<Rect> faces;
	int i;

	face_classifier.load("haarcascade_frontalface_alt.xml");

	while (1)
	{
		if (cap.grab() == 0) break;
		cap.retrieve(frame);
		cvtColor(frame, grayframe, CV_BGR2GRAY);
		equalizeHist(grayframe, grayframe);

		face_classifier.detectMultiScale(grayframe, faces, 1.1, 7, 0, Size(50, 50));

		for (i = 0; i < faces.size(); i++)
		{
			Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Point tr(faces[i].x, faces[i].y);
			rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 4, 0);
		}

		imshow("result", frame);
		waitKey(33);
	}
	*/



	// 1.2  Face_Detection_backward
	/*
	CascadeClassifier face_classifier;
	VideoCapture cap("face.mp4");
	Mat frame, grayframe;
	vector<Rect> faces;
	int i;

	face_classifier.load("haarcascade_frontalface_alt.xml");

	while (1)
	{
		if (cap.grab() == 0) break;
		cap.retrieve(frame);
		cvtColor(frame, grayframe, CV_BGR2GRAY);
		equalizeHist(grayframe, grayframe);

		face_classifier.detectMultiScale(grayframe, faces, 1.04, 2, 0, Size(0, 0), Size(40, 40));

		for (i = 0; i < faces.size(); i++)
		{
			Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Point tr(faces[i].x, faces[i].y);
			rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 4, 0);
		}
		imshow("result", frame);
		waitKey(33);
	}
	*/



	// 2.1 Face_Tracking_forward
	/*
	CascadeClassifier face_classifier;
	VideoCapture cap("face.mp4");
	Mat frame, grayframe, hsv, m_backproj;
	vector<Rect> faces;
	int i;
	int cur_frame;
	Rect rect2;

	MatND m_model3d;

	float hrange[] = { 0,180 };

	float vrange[] = { 0,255 };

	const float* ranges[] = { hrange, vrange, vrange };	// hue, saturation, brightness

	int channels[] = { 0, 1, 2 };

	int hist_sizes[] = { 16, 16, 16 };

	face_classifier.load("haarcascade_frontalface_alt.xml");

	bool flag = false;

	int height, width;

	while (1)
	{
		if (cap.grab() == 0) break;
		cap.retrieve(frame);
		cvtColor(frame, hsv, CV_BGR2HSV);

		cur_frame = cap.get(CAP_PROP_POS_FRAMES);

		cvtColor(frame, grayframe, CV_BGR2GRAY);
		equalizeHist(grayframe, grayframe);

		face_classifier.detectMultiScale(grayframe, faces, 1.1, 7, 0, Size(50, 50));

		if (cur_frame % 10 == 1 and faces.size() != 0)
		{
			Point lb(faces[0].x + faces[0].width, faces[0].y + faces[0].height);
			Point tr(faces[0].x, faces[0].y);

			height = faces[0].height;
			width = faces[0].width;

			Rect rect(lb, tr);
			rect2 = rect;

			Mat mask = Mat::zeros(height, width, CV_8U);

			ellipse(mask, Point(width / 2, height / 2), Size(width / 2, height / 2), 0, 0, 360, 255, CV_FILLED);

			Mat roi(hsv, rect);
			calcHist(&roi, 1, channels, mask, m_model3d, 3, hist_sizes, ranges);

		}

		calcBackProject(&hsv, 1, channels, m_model3d, m_backproj, ranges);

		CamShift(m_backproj, rect2, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 1));

		rectangle(frame, rect2, Scalar(0, 255, 0), 3, CV_AA);

		imshow("result", frame);
		waitKey(33);
	}
	*/


	// 2.2 Face_Tracking_backward
	/*
	CascadeClassifier face_classifier;
	VideoCapture cap("face.mp4");
	Mat frame, grayframe, hsv, m_backproj;
	vector<Rect> faces;
	int i;
	int cur_frame;
	Rect rect2;

	MatND m_model3d;

	float hrange[] = { 0,180 };

	float vrange[] = { 0,255 };

	const float* ranges[] = { hrange, vrange, vrange };	// hue, saturation, brightness

	int channels[] = { 0, 1, 2 };

	int hist_sizes[] = { 16, 16, 16 };

	face_classifier.load("haarcascade_frontalface_alt.xml");

	bool flag = false;

	int height, width;

	while (1)
	{
		if (cap.grab() == 0) break;
		cap.retrieve(frame);
		cvtColor(frame, hsv, CV_BGR2HSV);

		cur_frame = cap.get(CAP_PROP_POS_FRAMES);

		cvtColor(frame, grayframe, CV_BGR2GRAY);
		equalizeHist(grayframe, grayframe);

		face_classifier.detectMultiScale(grayframe, faces, 1.04, 6, 0, Size(0, 0), Size(40, 40));

		if (cur_frame % 2 == 1 and faces.size() != 0)
		{
			Point lb(faces[0].x + faces[0].width, faces[0].y + faces[0].height);
			Point tr(faces[0].x, faces[0].y);

			height = faces[0].height;
			width = faces[0].width;

			Rect rect(lb, tr);
			rect2 = rect;

			Mat mask = Mat::zeros(height, width, CV_8U);

			ellipse(mask, Point(width / 2, height / 2), Size(width / 2, height / 2), 0, 0, 360, 255, CV_FILLED);

			Mat roi(hsv, rect);
			calcHist(&roi, 1, channels, mask, m_model3d, 3, hist_sizes, ranges);

		}

		calcBackProject(&hsv, 1, channels, m_model3d, m_backproj, ranges);

		CamShift(m_backproj, rect2, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 1));

		rectangle(frame, rect2, Scalar(0, 255, 0), 3, CV_AA);

		imshow("result", frame);
		waitKey(33);
	}
	*/	

	waitKey(0);
	return 0;
	}


