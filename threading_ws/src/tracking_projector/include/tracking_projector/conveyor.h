#ifndef CONVEYOR_H
#define CONVEYOR_H

#include <QPixmap>
#include "cherry_cpp.hpp"
#include <QPainter>
#include <QPaintDevice>
#include <QColor>
#include "frame.h"
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>

class Conveyor
{
public:
    Conveyor();
    void Add(Frame frame);

    // objecto holding all the frames
    std::vector<Frame> Frames;

    // settings
    void SetPitVisibility(bool visibility);
    void SetCleanVisibility(bool visibility);
    void SetMaybeVisibility(bool visibility);
    void SetSideVisibility(bool visibility);

    void SetScreenWidth(int value);   // mm
    void SetMountAngle(double value); // radians
    void SetOffsetX(int value);       // mm
    void SetOffsetY(int value);       // mm

    void SetMapPolyA(double value); 
    void SetMapPolyB(double value); 
    void SetMapPolyC(double value);
    double GetMapPolyA(); 
    double GetMapPolyB(); 
    double GetMapPolyC(); 

    bool GetPitVisibility();
    bool GetCleanVisibility();
    bool GetMaybeVisibility();
    bool GetSideVisibility();

    int GetScreenWidth();   // mm
    double GetMountAngle(); // radians
    int GetOffsetX();       // mm
    int GetOffsetY();       // mm

    void Redraw();
    void CalcualteMaps();

    void SetFrameCallback(std::function<void(QImage conveyor, QImage projector, long reference_count)>);

    // image to move around/draw
    QImage getPixmap(int encoderCount);
    QImage getPixmapWarped(int encoderCount);
    std::mutex pixmap_mutex;

private:
    bool pitVisible = false;
    bool cleanVisible = false;
    bool sideVisible = false;
    bool maybeVisible = true;

    double screenWidth = 2154;
    double mountAngle = 10 * M_PI / 180; // 10 deg in radians
    double offsetY = 110;
    double offsetX = 3050;
    double mapPolyA_, mapPolyB_, mapPolyC_;
    int mapMethod_ = 1;

    cv::Mat map_x;
    cv::Mat map_y;

    // QPixmap maybes;
    // QPixmap pits;
    // QPixmap sides;
    // QPixmap cleans;

    QImage framesPixmap = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
    QImage distoredPixmap = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
    QImage conveyor = QImage(3840, 1080, QImage::Format_ARGB32_Premultiplied);
    QImage screen = QImage(1920, 1080, QImage::Format_ARGB32_Premultiplied);
    long encoderReference = 0;

    QImage maybes = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
    QImage pits = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
    QImage sides = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
    QImage cleans = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);

    // QPixmap maybes = QPixmap(6000, 1080);
    // QPixmap pits = QPixmap(6000, 1080);
    // QPixmap sides = QPixmap(6000, 1080);
    // QPixmap cleans = QPixmap(6000, 1080);

    // QPixmap oldMaybes = QPixmap(6000, 1080);
    // QPixmap oldPits = QPixmap(6000, 1080);
    // QPixmap oldSides = QPixmap(6000, 1080);
    // QPixmap oldCleans = QPixmap(6000, 1080);

    void drawAll();
    void drawDistorted();
    void drawLayers(long refernce);

    void calculateMapsTrig();
    void calculateMapsPoly();
    // void clear();

    double y_factor(double y) { return (
                                           -y * y * 0.000176969104130805) +
                                       (y * 1.23068934149292) +
                                       15.7294704397464; }
    double y_factor_inverse(double y) { return (
                                                   y * y * 0.0001382215105681) +
                                               (y * 0.7902422197254557) +
                                               10.8900679064572; }

    void purgeOld(int currentEncoderCount);

    std::thread drawThread;
};

#endif // CONVEYOR_H