#include "frame.h"
#include <thread>

// Frame::pitColor = QColor.magenta;
// Frame::cleanColor = QColor.green;
// Frame::maybeColor = QColor.yellow;
// Frame::sideColor = QColor.cyan;

// Frame::width = 200;
// Frame::height = 1000;
// Frame::circle_size = 50;
int Frame::circle_size = 50;


Frame::Frame()
{
    this->pitColor = Qt::magenta;
    this->cleanColor = Qt::green;
    this->maybeColor = Qt::yellow;
    this->sideColor = Qt::cyan;
    this->encoderCount = 0;
    // this->cherries = cherries;
    this->height = 1080;
    this->width = 400;
    // this->circle_size = 25;

    // initialize / clear the Pixmap
    pits = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    maybes = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    cleans = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    sides = QImage(width, height, QImage::Format_ARGB32_Premultiplied);

    // draw 4 pixmaps based on each type.
    drawAll();
};

Frame::Frame(
    std::vector<Cherry_cpp> cherries,
    int encoderCount,
    QColor pitColor = Qt::magenta,
    QColor cleanColor = Qt::green,
    QColor maybeColor = Qt::yellow,
    QColor sideColor = Qt::cyan,
    int width = 400,
    int height = 1080)
{

    // note the endcoder count & cherries
    this->pitColor = pitColor;
    this->cleanColor = cleanColor;
    this->maybeColor = maybeColor;
    this->sideColor = sideColor;
    this->encoderCount = encoderCount;
    this->cherries = cherries;
    this->height = height;
    this->width = width;
    // this->circle_size = circle_size;

    // initialize / clear the Pixmap
    pits = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    maybes = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    cleans = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    sides = QImage(width, height, QImage::Format_ARGB32_Premultiplied);

    // draw 4 pixmaps based on each type.
    drawAll();
};

Frame::Frame(
    std::vector<Cherry_cpp> cherries,
    int encoderCount)
{
    this->encoderCount = encoderCount;
    this->cherries = cherries;
    this->pitColor = Qt::magenta;
    this->cleanColor = Qt::green;
    this->maybeColor = Qt::yellow;
    this->sideColor = Qt::cyan;
    // this->cherries = cherries;
    this->height = 1080;
    this->width = 400;
    // this->circle_size = 25;

    // initialize / clear the Pixmap
    pits = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    maybes = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    cleans = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    sides = QImage(width, height, QImage::Format_ARGB32_Premultiplied);

    // draw 4 pixmaps based on each type.
    std::thread drawThread(&Frame::drawAll, this);

    drawThread.join();
};

// void Frame::clear()
// {
//     // initialize pixelmaps
//     // pits = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
//     pits = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
//     maybes = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
//     cleans = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
//     sides = QImage(width, height, QImage::Format_ARGB32_Premultiplied);
// };

void Frame::drawAll()
{

    // macke backgrounds transparent
    pits.fill(Qt::transparent);
    maybes.fill(Qt::transparent);
    cleans.fill(Qt::transparent);
    sides.fill(Qt::transparent);

    QPainter pitPainter = QPainter(&pits);
    QPainter maybePainter = QPainter(&maybes);
    QPainter cleanPainter = QPainter(&cleans);
    QPainter sidePainter = QPainter(&sides);

    QBrush circleBrush_ok = QBrush(Qt::green);
    QBrush circleBrush_ng = QBrush(Qt::magenta);
    QBrush circleBrush_side = QBrush(Qt::cyan);
    QBrush circleBrush_maybe = QBrush(Qt::yellow);
    QPen circlePen = QPen(Qt::black);
    circlePen.setWidth(1);

    pitPainter.setBrush(circleBrush_ng);
    maybePainter.setBrush(circleBrush_maybe);
    sidePainter.setBrush(circleBrush_side);
    cleanPainter.setBrush(circleBrush_ok);

    pitPainter.setPen(circlePen);
    maybePainter.setPen(circlePen);
    sidePainter.setPen(circlePen);
    cleanPainter.setPen(circlePen);

    int circleRadius = circle_size / 2;
    int mid_x = width / 2;
    int mid_y = height / 2;

    // draw each cherry
    for (int argi = 0; argi < cherries.size(); argi++)
    {
        Cherry_cpp cherry = cherries[argi];
        int x = cherry.X; // + mid_x;
        int y = height-cherry.Y; // + mid_y;

        if (cherries[argi].Type == 1)
        {
            cleanPainter.drawEllipse(QPointF(x, y),
                                     circleRadius, circleRadius);
        }
        else if (cherries[argi].Type == 3)
        {
            sidePainter.drawEllipse(QPointF(x, y),
                                    circleRadius, circleRadius);
        }
        else if (cherries[argi].Type == 2)
        {
            pitPainter.drawEllipse(QPointF(x, y),
                                   circleRadius, circleRadius);
        }
        else if (cherries[argi].Type == 5)
        {
            maybePainter.drawEllipse(QPointF(x, y),
                                     circleRadius, circleRadius);
        }
    }

    // pits.save("/home/wesley/Pictures/Screenshots/pits.png");
};

void Frame::SetCircleSize(int value)
{
    circle_size = value;
};

int Frame::GetCircleSize(){
    return circle_size;
}


QImage Frame::getSides()
{
    return sides;
};

QImage Frame::getCleans()
{
    return cleans;
};

QImage Frame::getMaybes()
{
    return maybes;
};

QImage Frame::getPits()
{
    return pits;
};

int Frame::getEncoderCount()
{
    return encoderCount;
}