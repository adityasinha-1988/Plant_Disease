#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <QFileDialog>
#include <iostream>
#include <QColor>

using namespace cv;
using namespace std;

int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
int lowThreshold;
int const max_BINARY_value = 255;
int threshold_type;

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
      Mat src_gray;
      Mat grad;
      int scale = 1;
      int delta = 0;
      int ddepth = CV_16S;

      GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

      /// Convert it to gray
      //cvtColor( src, src_gray, CV_BGR2GRAY );
      src_gray=dst.clone();
      cout<<dst.rows<<endl;

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

      ui->label_2->setPixmap(QPixmap::fromImage(QImage(grad.data, grad.cols, grad.rows, grad.step, QImage::Format_Mono)));

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
}

void::MainWindow::gray()
{
    cvtColor( src, src_gray, CV_BGR2GRAY );
    ui->label->setPixmap(QPixmap::fromImage(QImage(src_gray.data, src_gray.cols, src_gray.rows, src_gray.step, QImage::Format_Indexed8)));
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
    cout<<"mean::"<<mean(0)<<endl<<"mean2::"<<mean<<endl<<"mean3::"<<mean(2)<<endl;
    //ui->lineEdit_2->setText(QString::from(mean(0)));

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
