#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <QFileDialog>
#include <iostream>
#include <QColor>
#include <glcm.h>

using namespace cv;
using namespace std;

int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
int lowThreshold;
int const max_BINARY_value = 255;
int threshold_type;
int color_type;
RNG rng(12345);
int thresh = 100;


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
   // connect(ui->pushButton,SIGNAL(clicked()),this,SLOT(contour()));
    connect(ui->pushButton,SIGNAL(clicked()),this,SLOT(open_img()));
    connect(ui->pushButton_2,SIGNAL(clicked()),this,SLOT(calc_histogram()));
    connect(ui->pushButton_3,SIGNAL(clicked()),this,SLOT(sobel()));
    connect(ui->pushButton_4,SIGNAL(clicked()),this,SLOT(calc_threshold()));
    connect(ui->pushButton_5,SIGNAL(clicked()),this,SLOT(gray()));
    connect(ui->pushButton_6,SIGNAL(clicked()),this,SLOT(HSV()));
    connect(ui->pushButton_8,SIGNAL(clicked()),this,SLOT(features()));
    connect(ui->dft_button,SIGNAL(clicked()),this,SLOT(n_dft_button_clicked()));
    connect(ui->pushButton_10,SIGNAL(clicked()),this,SLOT(on_pushButton_10_clicked()));
    connect(ui->pushButton_11,SIGNAL(clicked()),this,SLOT(greyLCM()));

}


MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::open_img()
{
    filename = QFileDialog::getOpenFileName(this, "Open File",
                                                    "Images (*.png *.xpm *jpeg *.jpg *.jpe *.jp2 *.bmp *.tiff *.tif *.dib *.pbm *.pgm *.ppm)");

    QImage image(filename);
    src=imread(filename.toStdString());
    ui->label->setPixmap(QPixmap::fromImage(image));
}


void MainWindow::calc_histogram()
{
        Mat b_hist, g_hist, r_hist;

    /// Separate the image in 3 places ( B, G and R )
        vector<Mat> bgr_planes;
        split( src , bgr_planes );

        /// Establish the number of bins
         int histSize = 256;

         /// Set the ranges ( for B,G,R) )
         float range[] = { 0, 256 } ;
         const float* histRange = { range };

         bool uniform = true; bool accumulate = false;

        /// Compute the histograms:
        calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

        // Draw the histograms for B, G and R
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );

        Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );


        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                           Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                           Scalar( 255, 0, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                           Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                           Scalar( 0, 255, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                           Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                           Scalar( 0, 0, 255), 2, 8, 0  );
        }

        ui->label_2->setPixmap(QPixmap::fromImage(QImage(histImage.data, histImage.cols, histImage.rows, histImage.step, QImage::Format_RGB888)));

}

void MainWindow::contour()
{
    Mat src_gray; Mat canny_output; Mat dst, detected_edges;
    QImage image("/home/aditya/Pictures/131.jpg");
    src = imread("/home/aditya/Pictures/131.jpg");
    cvtColor( src, src_gray, CV_BGR2GRAY );
    dst.create( src.size(), src.type() );
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    dst = Scalar::all(0);
    src.copyTo( dst, detected_edges);
    cout<<dst.rows;
    QImage img;
   // img.fromData(dst.,dst.cols,dst.rows);
    
   // ui->label->setPixmap(QPixmap::loadFromData(dst.data),dst.cols, dst.rows, dst.step, QImage::Format_RGB888);
}

void MainWindow::sobel()
{
      Mat grad;
      int scale = 1;
      int delta = 0;
      int ddepth = CV_16S;

      GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
     // cv::cvtColor(src,src,)

      /// Convert it to gray
      //cvtColor( src, src_gray, CV_BGR2GRAY );
      //cout<<src.rows<<endl<<src_gray.rows<<endl;

      /// Generate grad_x and grad_y
      Mat grad_x, grad_y;
      Mat abs_grad_x, abs_grad_y;

      /// Gradient X
      //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
      Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
      convertScaleAbs( grad_x, abs_grad_x );

      /// Gradient Y
      //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
      Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
      convertScaleAbs( grad_y, abs_grad_y );

      /// Total Gradient (approximate)
      addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

      ui->label_2->setPixmap(QPixmap::fromImage(QImage(grad.data, grad.cols, grad.rows, grad.step, QImage::Format_Indexed8)));
      grad.copyTo(src_gray);
}

void::MainWindow::calc_threshold()
{
    Mat dst;
    ui->horizontalScrollBar->setEnabled(true);
    ui->verticalScrollBar->setEnabled(true);
    ui->radioButton->setEnabled(true);
    ui->radioButton_2->setEnabled(true);
    ui->radioButton_3->setEnabled(true);
    ui->radioButton_4->setEnabled(true);
    ui->radioButton_5->setEnabled(true);

    threshold( src_gray, dst, ui->horizontalScrollBar->value(), max_BINARY_value,threshold_type);
    ui->label_2->setPixmap(QPixmap::fromImage(QImage(dst.data, dst.cols, dst.rows, dst.step, QImage::Format_Indexed8)));
    dst.copyTo(src_gray);
}

void::MainWindow::gray()
{
    cvtColor( src, src_gray, CV_BGR2GRAY );
    ui->label->setPixmap(QPixmap::fromImage(QImage(src_gray.data, src_gray.cols, src_gray.rows, src_gray.step, QImage::Format_Indexed8)));

    ui->pushButton_3->setEnabled(true);
    ui->pushButton_4->setEnabled(true);
    ui->dft_button->setEnabled(true);
}


void MainWindow::on_radioButton_clicked()
{
    threshold_type=0;
}

void MainWindow::on_radioButton_2_clicked()
{
    threshold_type=1;
}

void MainWindow::on_radioButton_3_clicked()
{
    threshold_type=2;
}

void MainWindow::on_radioButton_4_clicked()
{
    threshold_type=3;
}

void MainWindow::on_radioButton_5_clicked()
{
    threshold_type=4;
}

void MainWindow::features()
{
    CvScalar MeanScalar;
    CvScalar Scalar1;
    //cvAvgSdv(src,&MeanScalar,&StandardDeviationScalar);

    std::vector<uchar> array;
    if (src_gray.isContinuous()) {
      array.assign(src_gray.datastart, src_gray.dataend);
    } else {
      for (int i = 0; i < src_gray.rows; ++i) {
        array.insert(array.end(), src_gray.ptr<uchar>(i), src_gray.ptr<uchar>(i)+src_gray.cols);
      }
    }

    Scalar mean, stddev;
    meanStdDev(src, mean, stddev);
   // ui->lineEdit->setText(mean[0]);
    cout<<"Blue channel mean::"<<mean(0)<<endl<<"Green channel mean::"<<mean(1)<<endl<<"Red channel mean::"<<mean(2)<<endl;

    ui->lineEdit->setText(QString::number(static_cast<int>(mean(0))));
    ui->lineEdit_2->setText(QString::number(static_cast<int>(stddev(0))));



     //Scalar1 = cvAvg(array.);

/*    Scalar mean=0;
    for(int i=0; i<src_gray.rows; i++)
        for(int j=0; j<src_gray.cols; j++)
            {
                mean = mean + src_gray.at<uchar>(i,j);
            }
    cout<<mean<<endl;


       printf("Blue Channel Avg is : %.f\n",MeanScalar.val[0]);
       printf("Blue Channel Standard Deviation is :%.f\n",StandardDeviationScalar.val[0]);
       printf("Green Channel Avg is : %.f\n",MeanScalar.val[1]);
       printf("Green Channel Standard Deviation is :%.f\n",StandardDeviationScalar.val[1]);
       printf("Red Channel Avg is : %.f\n",MeanScalar.val[2]);
       printf("Red Channel Standard Deviation is :%.f\n",StandardDeviationScalar.val[2]);*/

        printf("Blue Channel Avg is : %.f\n",Scalar1.val[0]);
        printf("Green Channel Avg is : %.f\n",Scalar1.val[1]);
        printf("Red Channel Avg is : %.f\n",Scalar1.val[2]);

}

void MainWindow::HSV()
{
    cvtColor( src, src_hsv, CV_BGR2HSV );
    ui->label->setPixmap(QPixmap::fromImage(QImage(src_hsv.data, src_hsv.cols, src_hsv.rows, src_hsv.step, QImage::Format_RGB888)));
    src_hsv.copyTo(src);

    imshow("HSV", src_hsv);

  //  QColor color = image.pixelColor();
  // int hue = color.hue();

    // modify hue as you’d like and write back to the image
   // color.setHsv(hue, color.saturation(), color.value(), color.alpha());
//    image.setPixelColor(i, j, color);
}

void MainWindow::on_pushButton_9_clicked()
{
    QImage image(filename);
    src=imread(filename.toStdString());
    ui->label->setPixmap(QPixmap::fromImage(image));
}


void MainWindow::on_dft_button_clicked()
{
    Mat I;
    src_gray.copyTo(I);
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);

    ui->label->setPixmap(QPixmap::fromImage(QImage(I.data, src_gray.cols, I.rows, I.step, QImage::Format_Indexed8)));
    //ui->label_5->setPixmap(QPixmap::fromImage(QImage(magI.data, magI.cols, magI.rows, magI.step, QImage::Format_Indexed8)));
}

Mat MainWindow::colorConversion()
{
     cvtColor(src, src_hsv, color_type);
     return src_img;
}

void MainWindow::on_pushButton_10_clicked()
{
    //cv::cvtColor(src, src_LAB, CV_BGR2Lab);
    //imshow("LAB", src_LAB);
   // cvtColor( src_LAB, src, CV_Lab2RGB );
    //imshow("newRGB", src);
    cvtColor( src, src_gray, CV_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) );

    threshold( src_gray, src_gray, 85, 255, 2);
    GaussianBlur( src_gray, src_gray, Size(3,3), 0, 0, BORDER_DEFAULT );
    imshow("trun", src_gray);

    char* source_window = "Source";
    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( source_window, src );
    //createTrackbar( " Threshold:", "Source", &thresh, 255, thresh_callback);
    //thresh_callback( 0, 0 );
}

void MainWindow::thresh_callback(int, void*)
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );

  /// Find contours
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Find the convex hull object for each contour
  vector<vector<Point> >hull( contours.size() );
  for( int i = 0; i < contours.size(); i++ )
     {  convexHull( Mat(contours[i]), hull[i], false ); }

  /// Draw contours + hull results
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
     }

  /// Show in a window
  namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
  imshow( "Hull demo", drawing );
}

void MainWindow::greyLCM()
{
        char key;
        GLCM glcm;
        TextureEValues EValues;

        // the Time Statistical Variable of Program Running Time
        double time;
        double start;

        // the Matrixs of Texture Features
        Mat imgEnergy, imgContrast, imgHomogenity, imgEntropy;

        // Read a Image
        //img = imread("/home/aditya/Downloads/GLCM-OpenCV-master/image/Satellite.jpg");

        Mat dstChannel;
        glcm.getOneChannel(src, dstChannel, CHANNEL_B);

        // Magnitude Gray Image, and calculate program running time
        start = static_cast<double>(getTickCount());
        glcm.GrayMagnitude(dstChannel, dstChannel, GRAY_8);
        time = ((double)getTickCount() - start) / getTickFrequency() * 1000;
        cout << "Time of Magnitude Gray Image: " << time << "ms" <<endl<<endl;

        // Calculate Texture Features of the whole Image, and calculate program running time
        start = static_cast<double>(getTickCount());
        glcm.CalcuTextureImages(dstChannel, imgEnergy, imgContrast, imgHomogenity, imgEntropy, 5, GRAY_8, true);
        time = ((double)getTickCount() - start) / getTickFrequency() * 1000;
        cout << "Time of Generate the whole Image's Calculate Texture Features Image: " << time << "ms" << endl<<endl;

        start = static_cast<double>(getTickCount());
        glcm.CalcuTextureEValue(dstChannel, EValues, 5, GRAY_8);
        time = ((double)getTickCount() - start) / getTickFrequency() * 1000;
        cout << "Time of Calculate Texture Features of the whole Image: " << time << "ms" << endl<<endl;

        cout<<"Image's Texture EValues:"<<endl;
        cout<<"    Contrast: "<<EValues.contrast<<endl;
        cout<<"    Energy: "<<EValues.energy<<endl;
        cout<<"    EntropyData: "<<EValues.entropy<<endl;
        cout<<"    Homogenity: "<<EValues.homogenity<<endl;

        ui->lineEdit_3->setText(QString::number(EValues.contrast));
        ui->lineEdit_6->setText(QString::number(EValues.entropy));
        ui->lineEdit_5->setText(QString::number(EValues.energy));
        ui->lineEdit_7->setText(QString::number(EValues.homogenity));

        FILE *fp;

        char a[10],b[10],c[10],d[10];
        sprintf(a, "%f", EValues.contrast);
        sprintf(b, "%f", EValues.energy);
        sprintf(c, "%f", EValues.entropy);
        sprintf(d, "%f", EValues.homogenity);

        fp = fopen("/home/aditya/Documents/test.txt", "a");
        fprintf(fp, " %s %s %s %s ", a, b, c, d);

    //fprintf(fp, EValues.contrast, EValues.energy, EValues.entropy, EValues.homogenity);
        fclose(fp);

        imshow("Energy", imgEnergy);
        imshow("Contrast", imgContrast);
        imshow("Homogenity", imgHomogenity);
        imshow("Entropy", imgEntropy);

        key = (char) cvWaitKey(0);
}
