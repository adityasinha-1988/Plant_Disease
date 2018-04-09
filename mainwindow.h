#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv/cv.h>
#include <glcm.h>


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();


private slots:

    void open_img();
    void contour();
    void calc_histogram();
    void sobel();
    void calc_threshold();
    void gray();
    //cv::Mat blur_img(cv::Mat);
    void features();
    void HSV();
    cv::Mat colorConversion();
    void greyLCM();

    void on_radioButton_clicked();
    void on_radioButton_2_clicked();
    void on_radioButton_3_clicked();
    void on_radioButton_4_clicked();
    void on_radioButton_5_clicked();

    void on_pushButton_9_clicked();

    void on_dft_button_clicked();

    void on_pushButton_10_clicked();


private:
    Ui::MainWindow *ui;
   // void blur_img();
    QString filename, imgPath;
    cv::Mat src, src_gray, src_hsv, dst, src_img;
    void thresh_callback(int, void* );
    //QSharedPointer<FindBeadsThread> findBeadsThread;

};

#endif // MAINWINDOW_H
