// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <filesystem>
#include <iostream>
using namespace cv;
using namespace std;
static void help(const char* programName)
{
	cout <<
		"\nA program using pyramid scaling, Canny, contours and contour simplification\n"
		"to find squares in a list of images (pic1-6.png)\n"
		"Returns sequence of squares detected on the image.\n"
		"Call:\n"
		"./" << programName << " [file_name (optional)]\n"
		"Using OpenCV version " << CV_VERSION << "\n" << endl;
}
int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";
// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}
// returns sequence of squares detected on the image.
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();
	Mat pyr, timg, gray0(image.size(), CV_8U), gray;
	// down-scale and upscale the image to filter out the noise
	pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, timg, image.size());
	vector<vector<Point> > contours;
	// find squares in every color plane of the image
	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);
		// try several threshold levels
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 0, thresh, 5);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// apply threshold if l!=0:
				// tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}
			// find contours and store them all as a list
			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
			vector<Point> approx;
			// test each contour
			for (size_t i = 0; i < contours.size(); i++)
			{
				// approximate contour with accuracy proportional
				// to the contour perimeter
				approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
				// square contours should have 4 vertices after approximation
				// relatively large area (to filter out noisy contours)
				// and be convex.
				// Note: absolute value of an area is used because
				// area may be positive or negative - in accordance with the
				// contour orientation
				if (approx.size() == 4 &&
					fabs(contourArea(approx)) > 1000 &&
					isContourConvex(approx))
				{
					double maxCosine = 0;
					for (int j = 2; j < 5; j++)
					{
						// find the maximum cosine of the angle between joint edges
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}
					// if cosines of all angles are small
					// (all angles are ~90 degree) then write quandrange
					// vertices to resultant sequence
					if (maxCosine < 0.3)
						squares.push_back(approx);
				}
			}
		}
	}
}

vector<vector<Point>> markerTemplates;

void loadMarkerTemplates() {

	std::string path = "numbers/";
	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		//std::cout << entry.path() << std::endl;

		Mat image = imread(entry.path().string(), 0);
		vector<Point> blackPoints;
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				if (image.at<unsigned char>(i, j) == 0) blackPoints.push_back(Point(j, i));
			}
		}
		markerTemplates.push_back(blackPoints);
	}

}

vector<int> retrieveMarkers(int markerMatrix[11][11]) {
	vector<int> result;
	for (int i = 0; i < markerTemplates.size(); i++) {
		bool ok = true;
		for (int j = 0; j < markerTemplates[i].size(); j++) {
			if (markerMatrix[markerTemplates[i][j].y + 3][markerTemplates[i][j].x + 3] == 1) {
				ok = false;
				break;
			}
		}
		if (ok) {
			result.push_back(i);
		}
	}
	return result;
}

void printMarkerNames(vector<int> indexes) {
	std::string path = "numbers/";
	int i = 0;
	vector<string> names;

	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		//std::cout << entry.path() << std::endl;
		names.push_back(entry.path().string());
	}

	for (int i = 0; i < indexes.size(); i++) {
		cout << names[indexes[i]] << endl;
	}
}

vector<int> processMarker(Mat image, vector<Point> contour) {
	Mat content;
	content.create(242, 242, CV_8UC1);
	Mat content2;

	vector<Point2f> objectPoints;
	//objectPoints.push_back(Point(0, 0));
	objectPoints.push_back(Point(44, 44));
	//objectPoints.push_back(Point(241, 0));
	objectPoints.push_back(Point(241 - 44, 44));
	//objectPoints.push_back(Point(241, 241));
	objectPoints.push_back(Point(241 - 44, 241 - 44));
	//objectPoints.push_back(Point(0, 241));
	objectPoints.push_back(Point(44, 241 - 44));

	Mat h;

	//cout << h << endl;

	//cout << endl;
	//cout << endl;

	h = findHomography(contour, objectPoints, RANSAC, 3);

	//cout << h << endl;

	warpPerspective(image, content, h, Size(242, 242));

	threshold(content, content, 128, 255, THRESH_OTSU);

	cvtColor(content, content2, COLOR_GRAY2RGB);

	int markerMatrix[11][11];

	for (int i = 0; i < 11; i++) {
		for (int j = 0; j < 11; j++) {
			//circle(content2, Point(11 + 22 * j, 11 + 22 * i), 3, Scalar(0, 0, 255), -1);
			markerMatrix[i][j] = content.at<unsigned char>(11 + 22 * i, 11 + 22 * j)/255;
			//printf("%d ", markerMatrix[i][j]);
		}
		//printf("\n");
	}
	//printf("\n");

	bool isValid = true;
	/*for (int i = 0; i < 11 && isValid; i++) {
		for (int j = 0; j < 11 && isValid; j++) {
			if (i == 0 || i == 1 || i == 9 || i == 10 || j == 0 || j == 1 || j == 9 || j == 10) {
				if (markerMatrix[i][j] == 1) isValid = false;
			}
		}
	}*/

	for (int i = 2; i < 9; i++) {
		if (markerMatrix[i][2] == 0) isValid = false;
		if (markerMatrix[i][8] == 0) isValid = false;
		if (markerMatrix[2][i] == 0) isValid = false;
		if (markerMatrix[8][i] == 0) isValid = false;
	}

	//printf("is valid: %d\n", isValid);

	if (!isValid) return vector<int>();

	vector<int> markers = retrieveMarkers(markerMatrix);

	//printMarkerNames(markers);

	//cout << "numero de marcadores encontrados com sucesso: " << markers.size() << endl;

	//imshow("content", content2);
	//waitKey(0);
	
	//return 0;

	return markers;
}

void test() {
	Mat lady = imread("lady.jpg", 1);
	
	Mat lady2 = lady.clone();

	for (int i = 0; i < lady2.rows; i++) {
		for (int j = 0; j < lady2.cols; j++) {
			Vec3b pixel = lady2.at<Vec3b>(i, j);
			int average = (pixel.val[0] + pixel.val[1] + pixel.val[2]) / 3;
			int limit = 20;
			if (abs(average - pixel.val[0]) > limit || abs(average - pixel.val[1]) > limit || abs(average - pixel.val[2]) > limit) {
				lady2.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			}
		}
	}
	
	imshow("lady", lady);
	imshow("lady2", lady2);
	waitKey(0);
}

int processMarker2(Mat image, vector<Point> contour) {
	Mat content;
	content.create(242, 242, CV_8UC1);
	Mat content2;

	vector<Point2f> objectPoints;
	//objectPoints.push_back(Point(0, 0));
	objectPoints.push_back(Point(44, 44));
	//objectPoints.push_back(Point(241, 0));
	objectPoints.push_back(Point(241-44, 44));
	//objectPoints.push_back(Point(241, 241));
	objectPoints.push_back(Point(241-44, 241-44));
	//objectPoints.push_back(Point(0, 241));
	objectPoints.push_back(Point(44, 241-44));

	Mat h;

	//cout << h << endl;

	//cout << endl;
	//cout << endl;

	h = findHomography(contour, objectPoints, RANSAC, 3);

	//cout << h << endl;

	warpPerspective(image, content, h, Size(242, 242));

	//threshold(content, content, 32, 255, THRESH_BINARY);

	//cvtColor(content, content2, COLOR_GRAY2RGB);

	

	int markerMatrix[11][11];

	for (int i = 0; i < 11; i++) {
		for (int j = 0; j < 11; j++) {
			circle(content2, Point(11 + 22 * j, 11 + 22 * i), 1, Scalar(0, 0, 255), -1);
			markerMatrix[i][j] = (content.at<unsigned char>(11 + 22 * i, 11 + 22 * j) > 64) ? 1 : 0;
			//printf("%d ", markerMatrix[i][j]);
		}
		//printf("\n");
	}
	//printf("\n");

	//imshow("debug", content2);
	//waitKey(1);

	/*for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			printf("%c",(markerMatrix[i+3][j+3]==0)?' ':'X');
		}
		printf("\n");
	}*/

	bool isValid = true;
	/*for (int i = 0; i < 11 && isValid; i++) {
		for (int j = 0; j < 11 && isValid; j++) {
			if (i == 0 || i == 1 || i == 9 || i == 10 || j == 0 || j == 1 || j == 9 || j == 10) {
				if (markerMatrix[i][j] == 1) isValid = false;
			}
		}
	}*/

	for (int i = 2; i < 9; i++) {
		if (markerMatrix[i][2] == 0) isValid = false;
		if (markerMatrix[i][8] == 0) isValid = false;
		if (markerMatrix[2][i] == 0) isValid = false;
		if (markerMatrix[8][i] == 0) isValid = false;
	}

	//printf(">>>>> %d\n", isValid);

	//printf("is valid: %d\n", isValid);

	if (!isValid) return -1;
	imshow("content", content);
	
	//else return 0;
	
	vector<int> markers = retrieveMarkers(markerMatrix);
	if (markers.size() > 0) return markers[0];

	//printMarkerNames(markers);

	//cout << "numero de marcadores encontrados com sucesso: " << markers.size() << endl;

	//imshow("content", content2);
	//waitKey(0);

	return -1;
}

vector<Point> orderContour(vector<Point> contour) {

	vector<Point> result;
	
	float xcenter = (contour[0].x + contour[1].x + contour[2].x + contour[3].x) / 4;
	float ycenter = (contour[0].y + contour[1].y + contour[2].y + contour[3].y) / 4;

	for (int i = 0; i < 4; i++) {
		if (contour[i].x < xcenter && contour[i].y < ycenter) {
			result.push_back(contour[i]);
			break;
		}
	}

	for (int i = 0; i < 4; i++) {
		if (contour[i].x > xcenter && contour[i].y < ycenter) {
			result.push_back(contour[i]);
			break;
		}
	}

	for (int i = 0; i < 4; i++) {
		if (contour[i].x > xcenter && contour[i].y > ycenter) {
			result.push_back(contour[i]);
			break;
		}
	}

	for (int i = 0; i < 4; i++) {
		if (contour[i].x < xcenter && contour[i].y > ycenter) {
			result.push_back(contour[i]);
			break;
		}
	}

	return result;
}

vector<Point> orderContour2(vector<Point> contour, int offset) {

	vector<Point> result;

	switch (offset%8) {
		case 0:
			result.push_back(contour[0]);
			result.push_back(contour[1]);
			result.push_back(contour[2]);
			result.push_back(contour[3]);
			break;
		case 1:
			result.push_back(contour[2]);
			result.push_back(contour[3]);
			result.push_back(contour[0]);
			result.push_back(contour[1]);
			break;
		case 2:
			result.push_back(contour[3]);
			result.push_back(contour[0]);
			result.push_back(contour[1]);
			result.push_back(contour[2]);
			break;
		case 3:
			result.push_back(contour[1]);
			result.push_back(contour[2]);
			result.push_back(contour[3]);
			result.push_back(contour[0]);
			break;

		default:
			result.push_back(contour[0]);
			result.push_back(contour[1]);
			result.push_back(contour[2]);
			result.push_back(contour[3]);
			break;
	}

	return result;
}

void boygirl_application() {

	// carregar as imagens do menino e da menina
	Mat boy_front = imread("boy_front.jpg");
	Mat boy_back = imread("boy_back.jpg");
	Mat girl_front = imread("girl_front.jpg");
	Mat girl_back = imread("girl_back.jpg");

	// inciar a camera
	VideoCapture capture;
	capture.open(0);
	//capture.open("1e2.avi");

	Mat frame, bin, gray, bin2, gray2;
	vector<vector<Point> > contours;

	Mat full;
	full.create(720, 1280, CV_8UC3);

	Mat full1 = full(Rect(0, 0, 640, 360));
	Mat full2 = full(Rect(640, 0, 640, 360));
	Mat full3 = full(Rect(0, 360, 640, 360));
	Mat full4 = full(Rect(640, 360, 640, 360));
	
	int counter = 0;

	// criar o loop principal da aplicacao
	while (true) {
		// ler o frame da camera
		capture.read(frame);
		resize(frame, frame, Size(640, 360));

		frame.copyTo(full1);

		cvtColor(frame, gray, COLOR_BGR2GRAY);

		// processar o marcador
		threshold(gray, bin, 64, 255, THRESH_BINARY);
		cvtColor(bin, bin2, COLOR_GRAY2BGR);
		bin2.copyTo(full2);

		Canny(bin, gray, 0, thresh, 5);
		cvtColor(gray, gray2, COLOR_GRAY2BGR);
		gray2.copyTo(full3);

		//imshow("bin", bin);
		//imshow("gray", gray);

		findContours(gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

		vector<Point> approx;

		//printf("%d\n", contours.size());

		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
			if (approx.size() == 4 &&
				fabs(contourArea(approx)) > 1000 &&
				isContourConvex(approx))
			{
				double maxCosine = 0;
				for (int j = 2; j < 5; j++)
				{
					double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
					maxCosine = MAX(maxCosine, cosine);
				}
				if (maxCosine < 0.3) {
					approx = orderContour(approx);
					if(approx.size() == 4) {
						int m = processMarker2(bin, approx);

						if (m >= 0) {
							printf("Marker ID: %d\n", m);

							vector<Point2f> objectPoints;
							objectPoints.push_back(Point(0, 0));
							objectPoints.push_back(Point(boy_front.cols, 0));
							objectPoints.push_back(Point(boy_front.cols, boy_front.rows));
							objectPoints.push_back(Point(0, boy_front.rows));

							approx = orderContour2(approx, m);

							Mat h = findHomography(objectPoints, approx, RANSAC, 3);

							Mat mini;

							switch (m) {
							case 0:
							case 1:
							case 2:
							case 3:mini = boy_front; break;
							case 4:
							case 5:
							case 6:
							case 7: mini = boy_back; break;
							case 8:
							case 9:
							case 10:
							case 11: mini = girl_front; break;
							default: mini = girl_back; break;
							}

							warpPerspective(mini, frame, h, Size(frame.cols, frame.rows), 1, BORDER_TRANSPARENT);
						}
					}

				}
			}
		}
		frame.copyTo(full4);

		// dependendo do ID do arquivo encontrado, atribuir uma ordem específica para os pontos do contorno
		// aplicar um warp especifico na imagem de saida	

		//exibe o resultado final
		//imshow("Front/Back sample application", frame);
		imshow("Front/Back sample application", full);
		waitKey(1);

		char filename[100];
		sprintf_s(filename, "output/%04d.jpg", counter++);
		imwrite(filename, full);
	}


}

void color_application() {

	// carregar as imagens do menino e da menina

	Mat marker4 = imread("marker4_.jpg", 1);
	Mat marker5 = imread("marker5_.jpg");

	Mat yellow;
	yellow.create(11, 11, CV_8UC3);
	Mat blue = yellow.clone();
	Mat green = yellow.clone();

	yellow = Scalar(0,255,255);
	blue = Scalar(255,0,0);
	green = Scalar(0,128,0);

	// inciar a camera
	VideoCapture capture;
	//capture.open(0);
	capture.open("4e5.avi");

	Mat frame, bin, gray, bin2, gray2;
	vector<vector<Point> > contours;

	Mat full;
	full.create(720, 1280, CV_8UC3);

	Mat full1 = full(Rect(0, 0, 640, 360));
	Mat full2 = full(Rect(640, 0, 640, 360));
	Mat full3 = full(Rect(0, 360, 640, 360));
	Mat full4 = full(Rect(640, 360, 640, 360));

	int counter = 0;

	// criar o loop principal da aplicacao
	while (true) {
		full = 0;
		
		// ler o frame da camera
		capture.read(frame);
		resize(frame, frame, Size(640, 360));

		frame.copyTo(full1);

		cvtColor(frame, gray, COLOR_BGR2GRAY);

		// processar o marcador
		threshold(gray, bin, 64, 255, THRESH_BINARY);
		//cvtColor(bin, bin2, COLOR_GRAY2BGR);
		//bin2.copyTo(full2);

		Canny(bin, gray, 0, thresh, 5);
		//cvtColor(gray, gray2, COLOR_GRAY2BGR);
		//gray2.copyTo(full3);

		//imshow("bin", bin);
		//imshow("gray", gray);

		findContours(gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

		vector<Point> approx;

		//printf("%d\n", contours.size());

		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
			if (approx.size() == 4 &&
				fabs(contourArea(approx)) > 1000 &&
				isContourConvex(approx))
			{
				double maxCosine = 0;
				for (int j = 2; j < 5; j++)
				{
					double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
					maxCosine = MAX(maxCosine, cosine);
				}
				if (maxCosine < 0.3) {
					approx = orderContour(approx);
					if (approx.size() == 4) {
						vector<int> ms = processMarker(bin, approx);

						vector<Point2f> objectPoints;
						objectPoints.push_back(Point(0, 0));
						objectPoints.push_back(Point(9, 0));
						objectPoints.push_back(Point(9, 9));
						objectPoints.push_back(Point(0, 9));

						Mat h = findHomography(objectPoints, approx, RANSAC, 3);

						for (int k = 0; k < ms.size(); k++) {
							printf("Marker ID [%d]: %d\n", k, ms[k]);
							if ((ms[k] / 8) == 0) {
								warpPerspective(yellow, frame, h, Size(frame.cols, frame.rows), 1, BORDER_TRANSPARENT);
								
								vector<Point2f> objectPoints2;
								objectPoints2.push_back(Point(20, 20));
								objectPoints2.push_back(Point(90, 20));
								objectPoints2.push_back(Point(90, 90));
								objectPoints2.push_back(Point(20, 90));

								Mat h2 = findHomography(objectPoints2, approx, RANSAC, 3);
								
								warpPerspective(marker4, full2, h2, Size(frame.cols, frame.rows), 1, BORDER_TRANSPARENT);

								/*for (int j = 0; j < 4; j++) {
									line(full2, approx[j], approx[(j + 1) % 4], Scalar(255, 255, 255), 1);
								}*/
							}
							else {
								warpPerspective(blue, frame, h, Size(frame.cols, frame.rows), 1, BORDER_TRANSPARENT);

								vector<Point2f> objectPoints2;
								objectPoints2.push_back(Point(20, 20));
								objectPoints2.push_back(Point(90, 20));
								objectPoints2.push_back(Point(90, 90));
								objectPoints2.push_back(Point(20, 90));

								Mat h2 = findHomography(objectPoints2, approx, RANSAC, 3);

								warpPerspective(marker5, full3, h2, Size(frame.cols, frame.rows), 1, BORDER_TRANSPARENT);

								/*for (int j = 0; j < 4; j++) {
									line(full3, approx[j], approx[(j + 1) % 4], Scalar(255, 255, 255), 1);
								}*/
							}
						}

						if ((ms.size() == 2)) {
							warpPerspective(green, frame, h, Size(frame.cols, frame.rows), 1, BORDER_TRANSPARENT);
						}
						printf("ms size: %d\n", ms.size());
					}

				}
			}
		}
		frame.copyTo(full4);

		// dependendo do ID do arquivo encontrado, atribuir uma ordem específica para os pontos do contorno
		// aplicar um warp especifico na imagem de saida	

		//exibe o resultado final
		//imshow("Front/Back sample application", frame);
		imshow("Color sample application", full);
		waitKey(1);

		char filename[100];
		sprintf_s(filename, "output/%04d.jpg", counter++);
		imwrite(filename, full);
	}


}




int main(int argc, char** argv)
{
	loadMarkerTemplates();
	boygirl_application();
	//color_application();
	//test();
	exit(0);
	
	
	loadMarkerTemplates();
	cout << markerTemplates.size() << endl;

	Mat image = imread("test2.jpg", 0);
	vector<vector<Point> > contours, squares;

	Mat image2, bin, gray;
	// image should contain the frame to be processed

	medianBlur(image, image2, 7);

	threshold(image2, bin, 128, 255, THRESH_OTSU);

	Canny(bin, gray, 0, thresh, 5);

	imshow("bin", bin);
	imshow("gray", gray);

	findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<Point> approx;

	printf("%d\n", contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
		if (approx.size() == 4 &&
			fabs(contourArea(approx)) > 1000 &&
			isContourConvex(approx))
		{
			double maxCosine = 0;
			for (int j = 2; j < 5; j++)
			{
				double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
				maxCosine = MAX(maxCosine, cosine);
			}
			if (maxCosine < 0.3) {
				squares.push_back(approx);
				processMarker(image, approx);
			}
		}
	}

	//printf("%d\n", squares.size());

	cvtColor(image, image, COLOR_GRAY2BGR);

	for (int i = 0; i < squares.size(); i++) {
		for (int j = 0; j < 4; j++) {
			line(image, squares[i][j], squares[i][(j + 1) % 4], Scalar(0, 0, 255), 3);
		}
		imshow("test", image);
		waitKey(0);
	}
	
	return 0;
}