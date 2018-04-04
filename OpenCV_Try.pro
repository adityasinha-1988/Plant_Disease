#-------------------------------------------------
#
# Project created by QtCreator 2017-08-05T11:27:56
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = OpenCV_Try
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

unix {
    CONFIG += link_pkgconfig
    PKGCONFIG += opencv
}

