# Driving-Line-Detection
주행 차선 인식 프로그램 //코드 수정중 버그 발생하여 수정중에 있음

### 사용환경
- windows10
- opencv4.40
- visual studio 2019

### 구현 과정
1) 영상 읽어오기
2) 노란색, 흰색 범위만 필터링하여 저장
3) BGR->GRAYSCALE 변환
4) Canny Edge Detection 과 Gaussian 필터로 엣지 추출
5) ROI 영역 지정을 통해 진행 방향 차선만 검술
6) Hough 변환으로 엣지에서의 직선 성분추출
7) 선형 회귀로 좌우 차선 직선만 검출
8) 자동차의 운행방향 예측
9) 영상에 차선을 더하고 진행방향 텍스트로 출력
10) 영상을 출력
