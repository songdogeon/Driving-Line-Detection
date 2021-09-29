#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

Mat img_color;

double trapBottomWidth = 0.85;
double trapTopWidth = 0.07;
double trapHeight = 0.4;
double imgSize, imgCenter;
double left_m, right_m;
Point left_b, right_b;
bool leftDetect = false, rightDetect = false;

Scalar lowerWhite = Scalar(200, 200, 200);
Scalar upperWhite = Scalar(255, 255, 255);
Scalar lowerYellow = Scalar(10, 100, 100);
Scalar upperYellow = Scalar(40, 255, 255);

Mat limitRegion(Mat imgEdge)
{
	//ROI 영역의 가장자리만 감지되도록 마스킹함
	//ROI 영역의 가장자리만 표시되는 이전 영상을 반환
	int width = imgEdge.cols; //열의 갯수 Y
	int height = imgEdge.rows; //행의 갯수 X 

	Mat output;
	Mat mask = Mat::zeros(height, width, CV_8UC1);

	//ROI 영역 정점 계산
	Point points[4]{
		Point((width * (1 - trapBottomWidth)) / 2, height),
		Point((width * (1 - trapTopWidth)) / 2, height - height * trapHeight),
		Point(width - (width * (1 - trapTopWidth)) / 2, height - height * trapHeight),
		Point(width - (width * (1 - trapBottomWidth)) / 2, height)
	};

	//정점으로 정의된 다각형 내부의 색상을 채운다
	fillConvexPoly(mask, points, 4, Scalar(255, 0, 0));

	bitwise_and(imgEdge, mask, output);

	return output;
}

vector<Vec4i> houghLines(Mat imgMask)
{
	vector<Vec4i> line;

	HoughLinesP(imgMask, line, 1, CV_PI / 180, 20, 10, 20);
	return line;
}

vector<vector<Vec4i>> separateLine(Mat imgEdge, vector<Vec4i> lines)
{
	//검출된 모든 허프 변환 직선들을 기울기별로 정렬 선의 기울기와 대략적인 위치에 따라 좌우로 분류
	vector<vector<Vec4i>> output(2);
	Point ini, fini;
	vector<double> slopes;
	vector<Vec4i> selectedLines, leftLines, rightLines;
	double slopeThresh = 0.3;

	//검출한 직선의 기울기를 계산
	for (int i = 0; i < lines.size(); i++)
	{
		Vec4i line = lines[i];
		ini = Point(line[0], line[1]);
		fini = Point(line[2], line[3]);

		double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y)) / (static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001);

		//수평에 가까운 기울기의 선은 제외
		if (abs(slope) > slopeThresh)
		{
			slopes.push_back(slope);
			selectedLines.push_back(line);
		}
	}

	imgCenter = static_cast<double>((imgEdge.cols / 2));
	for (int i = 0; i < selectedLines.size(); i++)
	{
		ini = Point(selectedLines[i][0], selectedLines[i][1]);
		fini = Point(selectedLines[i][2], selectedLines[i][3]);

		if (slopes[i] > 0 && fini.x > imgCenter && ini.x > imgCenter)
		{
			rightLines.push_back(selectedLines[i]);
			rightDetect = true;
		}
		else if (slopes[i] < 0 && fini.x < imgCenter && ini.x < imgCenter)
		{
			leftLines.push_back(selectedLines[i]);
			leftDetect = true;
		}
	}

	output[0] = rightLines;
	output[1] = leftLines;
	return output;
}

vector<Point> regression(vector<vector<Vec4i>> separatedLines, Mat imgInput)
{
	//선형 회귀를 통해 좌우 차선의 적합한 선을 찾음
	vector<Point> output(4);
	Point ini, fini;
	Point ini2, fini2;
	Vec4d left_line, right_line;
	vector<Point> left_pts, right_pts;

	if (rightDetect)
	{
		for (auto i : separatedLines[0])
		{
			ini = Point(i[0], i[1]);
			fini = Point(i[2], i[3]);

			right_pts.push_back(ini);
			right_pts.push_back(fini);
		}
		if (right_pts.size() > 0)
		{
			//주어진 contour에 최적화된 직선을 추출함
			fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01);

			right_m = right_line[1] / right_line[0]; //기울기
			right_b = Point(right_line[2], right_line[3]);
		}
	}
	if (leftDetect)
	{
		for (auto j : separatedLines[1])
		{
			ini = Point(j[0], j[1]);
			fini = Point(j[2], j[3]);

			left_pts.push_back(ini2);
			left_pts.push_back(fini2);
		}
		if (left_pts.size() > 0)
		{
			//주어진 contour에 최적화된 직선을 추출함
			fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0]; //기울기
			left_b = Point(left_line[2], left_line[3]);
		}
	}
	//좌우 선 각각의 두 점을 계산함
	//y = mx+b -> x = (y-b)/m
	int ini_y = imgInput.rows;
	int fini_y = 470;

	double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
	double right_fini_x = ((fini_y - right_b.y) / right_m) + right_b.x;

	double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
	double left_fini_x = ((fini_y - left_b.y) / left_m) + left_b.x;

	output[0] = Point(right_ini_x, ini_y);
	output[1] = Point(right_fini_x, fini_y);
	output[2] = Point(left_ini_x, ini_y);
	output[3] = Point(left_fini_x, fini_y);

	return output;
}

Mat drawLine(Mat imgInput, vector<Point> lane, string dir)
{
	// 차선을 경계로하는 다각형을 투명한 색으로 채움 
	// 좌우 차선을 선으로 그림
	// 예측 진행 방향 텍스트를 출력함.
	vector<Point> polyPoints;
	Mat output;
	imgInput.copyTo(output);
	polyPoints.push_back(lane[2]);
	polyPoints.push_back(lane[0]);
	polyPoints.push_back(lane[1]);
	polyPoints.push_back(lane[3]);

	fillConvexPoly(output, polyPoints, Scalar(0, 230, 30), LINE_AA, 0);
	addWeighted(output, 0.3, imgInput, 0.7, 0, imgInput);

	line(imgInput, lane[0], lane[1], Scalar(0, 255, 255), 5, LINE_AA);
	line(imgInput, lane[2], lane[3], Scalar(0, 255, 255), 5, LINE_AA);

	putText(imgInput, dir, Point(520, 100), FONT_HERSHEY_PLAIN, 3, Scalar(255, 255, 255), 3, LINE_AA);

	return imgInput;
}


string predictDir()
{
	//두 차선이 교차하는 지점이 중심점으로 왼쪽or오른쪽 인지 구분하여 진행방향예측
	string output;
	double vx;
	double thres_vp = 10;

	// 두차전이 교차하는 지점 계산
	vx = static_cast<double>(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

	if (vx < imgCenter - thres_vp)
		output = "좌회전";
	else if (vx > imgCenter + thres_vp)
		output = "우회전";
	else if (vx >= (imgCenter - thres_vp) && vx <= (imgCenter + thres_vp))
		output = "직진";

	return output;
}

void filterColors(Mat _imgBgr, Mat& imgFiltered)
{
	Mat imgBgr;
	_imgBgr.copyTo(imgBgr);
	Mat imgHsv, imgCombine;
	Mat whiteMask, whiteImage;
	Mat yellowMask, yellowImage;

	inRange(imgBgr, lowerWhite, upperWhite, whiteMask); //imgBgr에서 lower,upperwhite범위에서 픽셀 검출후whiteMask에저장
	bitwise_and(imgBgr, imgBgr, whiteImage, whiteMask); //파라미터 1,2에서 whilte mask를 검출후 white Image에 저장

	cvtColor(imgBgr, imgHsv, COLOR_BGR2HSV);

	inRange(imgHsv, lowerYellow, upperYellow, yellowMask);
	bitwise_and(imgBgr, imgBgr, yellowImage, yellowMask);

	addWeighted(whiteImage, 1.0, yellowImage, 1.0, 0.0, imgCombine);//파라미터: 2)알파 값, 4)베타 값,

	imgCombine.copyTo(imgFiltered);
}

Mat regionOfInterest(Mat imgEdge, Point* points)
{
	Mat imgMask = Mat::zeros(imgEdge.rows, imgEdge.cols, CV_8UC1); //CV_8UC1 타입으로 imgEdge원소를 초기화

	Scalar ignoreMaskColor = Scalar(255, 255, 255);
	const Point* ppt[1] = { points };
	int npt[] = { 4 };
	//타겟이미지, const Point**, npts, ncotours, 다각형 색상, 다각형 라인 타입
	fillPoly(imgMask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);//영역 속에 색을 채움

	Mat imgMasked;
	bitwise_and(imgEdge, imgMask, imgMasked);

	return imgMasked;
}

int main()
{
	Mat imgBgr, imgGray, imgEdge, imgMask, imgFrame, imgLines, imgResult;
	vector<Vec4i> lines;//int 형 4개를 저장할수 있는 벡터
	vector<vector<Vec4i>> separatedLines;
	vector<Point> lane;
	string dir;
	string videoFile = "C:\\opencv\\challenge.mp4";
	VideoCapture cap(videoFile);
	if (!cap.isOpened())
	{
		cout << "에러: 영상을 열 수 없습니다.";
	}

	while (1)
	{
		cap.read(imgFrame);
		if (imgFrame.empty()) break;

		imshow("Color", imgFrame);

		Mat imgFiltered;
		filterColors(imgFrame, imgFiltered);

		imshow("Filtered", imgFiltered);
		waitKey(25);

		cvtColor(imgFiltered, imgGray, COLOR_BGR2GRAY);
		GaussianBlur(imgGray, imgGray, Size(3, 3), 0, 0);
		Canny(imgGray, imgEdge, 50, 150);

		//imshow("Gray", imgGray);

		//imshow("Edge", imgEdge);

		imgMask = limitRegion(imgEdge);

		imshow("Masked", imgMask);

		//HoughLinesP(src,dst,rho,theta,threshold,min_line_length,max_line_gap)
		/*
		src:입력할 이미지 변수, edge detect 된 이미지를 입력해야함
		dst:허프변환 직선 검출 정보를 저장할 Array
		rho:계산할 픽셀(매개 변수)의 해상도, 그냥1을 사용하면됨 (변환된 그래프에서, 선에서 원점까지의 수직거리)
		theta:계산할 각도(라디안,매개변수)의 해상도, 선회전 각도 (모든 방향에서 직선을 검출하려면 PI/180을 사용)
		threshold:변환된 그래프에서 라인을 검출하기 위한 최소 교차 수
		min_line_length:검출할 직선의 최소 길이, 단위는 픽셀
		max_line_gap:검출할 선위의 점들 사이의 최대 거리, 점사이의 거리가 이값보다 크면 다른선으로 간주
		*/
		
		if (lines.size() > 0)
		{
			separatedLines = separateLine(imgMask, lines);
			lane = regression(separatedLines, imgFrame);

			dir = predictDir();

			imgResult = drawLine(imgFrame, lane, dir);
		}
		imshow("result", imgResult);

		if (waitKey(25) >= 0) break;
	}
	destroyAllWindows();

	return 0;
}