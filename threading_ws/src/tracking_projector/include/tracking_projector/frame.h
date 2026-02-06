#ifndef FRAME_H
#define FRAME_H

#include <QImage>
#include "cherry_cpp.hpp"
#include <QPainter>
#include <QColor>

class Frame
{
public:
    Frame();
    Frame(
        std::vector<Cherry_cpp> cherries,
        int encoderCount);
    Frame(
        std::vector<Cherry_cpp> cherries,
        int encoderCount,
        QColor pitColor,
        QColor cleanColor,
        QColor maybeColor,
        QColor sideColor,
        int width,
        int height);

    QImage getPits();   // = QImage(200, 1080);
    QImage getMaybes(); // = QImage(200, 1080);
    QImage getCleans(); // = QImage(200, 1080);
    QImage getSides();  // = QImage(200, 1080);
    int getEncoderCount();
    std::vector<Cherry_cpp> getCherries();

    static void SetCircleSize(int value);
    static int GetCircleSize();

private:
    QImage pits;   // = QImage(200, 1080);
    QImage maybes; // = QImage(200, 1080);
    QImage cleans; // = QImage(200, 1080);
    QImage sides;

    QColor pitColor;
    QColor cleanColor;
    QColor maybeColor;
    QColor sideColor;
    static int circle_size;

    // QPainter *pitPainter;
    // QPainter *maybePainter;
    // QPainter *cleanPainter;
    // QPainter *sidePainter;

    int encoderCount;
    std::vector<Cherry_cpp> cherries;

private:
    // Pixmap drawMaybe();
    // QImage drawClean();
    // QImage drawPits();
    void drawAll();
    // void clear();

    int width;
    int height;

    int bg_color_ = 0;
};

#endif // FRAME_H
