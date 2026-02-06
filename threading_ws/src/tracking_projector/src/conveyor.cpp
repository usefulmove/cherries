#include "conveyor.h"
#include <QDebug>

cv::Mat qimage_to_mat_cpy(QImage const &img)
{
    cv::Mat mat(img.height(), img.width(), CV_8UC4,
                const_cast<uchar *>(img.bits()),
                img.bytesPerLine());

    return mat.clone();
}

QImage mat_to_qimage_cpy(cv::Mat const &inMat)
{
    QImage image(inMat.data,
                 inMat.cols, inMat.rows,
                 static_cast<int>(inMat.step),
                 QImage::Format_ARGB32_Premultiplied);

    return image;
}

Conveyor::Conveyor()
{
    CalcualteMaps();
}

void Conveyor::SetPitVisibility(bool visibility)
{
    pitVisible = visibility;
}
void Conveyor::SetCleanVisibility(bool visibility)
{
    cleanVisible = visibility;
}
void Conveyor::SetMaybeVisibility(bool visibility)
{
    maybeVisible = visibility;
}
void Conveyor::SetSideVisibility(bool visibility)
{
    sideVisible = visibility;
}

void Conveyor::SetScreenWidth(int value)
{
    screenWidth = value;
    CalcualteMaps();
}

void Conveyor::SetMountAngle(double value)
{
    mountAngle = value;
    CalcualteMaps();
}

void Conveyor::SetOffsetX(int value)
{
    offsetX = value;
    CalcualteMaps();
}
void Conveyor::SetOffsetY(int value)
{
    offsetY = value;
    CalcualteMaps();
}

bool Conveyor::GetPitVisibility()
{
    return pitVisible;
}
bool Conveyor::GetCleanVisibility()
{
    return cleanVisible;
}
bool Conveyor::GetMaybeVisibility()
{
    return maybeVisible;
}
bool Conveyor::GetSideVisibility()
{
    return sideVisible;
}

int Conveyor::GetScreenWidth()
{
    return screenWidth;
}

double Conveyor::GetMountAngle()
{
    return mountAngle;
}

int Conveyor::GetOffsetX()
{
    return offsetX;
}
int Conveyor::GetOffsetY()
{
    return offsetY;
}


void Conveyor::SetMapPolyA(double value)
{
    mapPolyA_ = value;
    CalcualteMaps();
}

double Conveyor::GetMapPolyA()
{
    return mapPolyA_;
}

void Conveyor::SetMapPolyB(double value)
{
    mapPolyB_ = value;
    CalcualteMaps();
}

double Conveyor::GetMapPolyB()
{
    return mapPolyB_;
}

void Conveyor::SetMapPolyC(double value)
{
    mapPolyC_ = value;
    CalcualteMaps();
}

double Conveyor::GetMapPolyC()
{
    return mapPolyC_;
}

void Conveyor::CalcualteMaps(){
    calculateMapsPoly();
}

QImage Conveyor::getPixmap(int encoderCount)
{

    int offset = encoderCount - encoderReference;
    int offset2 = 1000;

    QRect screen = QRect(-offset + offset2, 0, 6000, 1080);

    std::lock_guard<std::mutex> guard(pixmap_mutex);
    return framesPixmap.copy(screen);
}

QImage Conveyor::getPixmapWarped(int encoderCount)
{

    int offset = (encoderCount - encoderReference) * 1920 / screenWidth;
    // convPainter.translate(-2800, 0);

    int offset2 = 1920 * 1000 / screenWidth;

    int offset3 = 200 * (1 - 1920.0 / screenWidth);

    QRect projectorScreen = QRect(offset2 - offset - offset3 - 300, 0, 2520, 1080);

    std::lock_guard<std::mutex> guard(pixmap_mutex);
    return distoredPixmap.copy(projectorScreen);
}

void Conveyor::Redraw()
{
    drawLayers(encoderReference);
}

// seperate this out to make it easier to change visibility of cherry types
void Conveyor::drawLayers(long reference)
{

    QImage all = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
    all.fill(Qt::transparent);
    QPainter allPainter = QPainter(&all);
    if (cleanVisible)
        allPainter.drawImage(0, 0, cleans);
    if (sideVisible)
        allPainter.drawImage(0, 0, sides);
    if (maybeVisible)
        allPainter.drawImage(0, 0, maybes);
    if (pitVisible)
        allPainter.drawImage(0, 0, pits);

    std::lock_guard<std::mutex> guard(pixmap_mutex);
    framesPixmap = all.copy();
    encoderReference = reference;

    try
    {
        drawDistorted();
    }
    catch (const std::exception &f)
    {
        qInfo() << f.what();
    }
}

void Conveyor::drawAll()
{
    try
    {
        maybes = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
        pits = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
        sides = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);
        cleans = QImage(7000, 1080, QImage::Format_ARGB32_Premultiplied);

        maybes.fill(Qt::transparent);
        pits.fill(Qt::transparent);
        sides.fill(Qt::transparent);
        cleans.fill(Qt::transparent);

        QPainter pitPainter = QPainter(&pits);
        QPainter maybePainter = QPainter(&maybes);
        QPainter cleanPainter = QPainter(&cleans);
        QPainter sidePainter = QPainter(&sides);

        int reference = Frames.back().getEncoderCount();

        for (int argi = 0; argi < Frames.size(); argi++)
        {
            Frame frame = Frames[argi];

            int offset = reference - frame.getEncoderCount() + 1000; // get the offset from the current position

            pitPainter.drawImage(
                offset,
                0,
                frame.getPits());
            maybePainter.drawImage(
                offset,
                0,
                frame.getMaybes());
            cleanPainter.drawImage(
                offset,
                0,
                frame.getCleans());
            sidePainter.drawImage(
                offset,
                0,
                frame.getSides());
        }

        drawLayers(reference);
        // if (frame_cb_)
        // {
        //     frame_cb_(getPixmap(reference), getPixmapWarped(reference), reference);
        // }
    }
    catch (const std::exception &e)
    {
        qInfo() << e.what();
    }
}

void Conveyor::calculateMapsTrig()
{
    cv::Mat img_source(framesPixmap.height(), framesPixmap.width(), CV_8UC4,
                       const_cast<uchar *>(framesPixmap.bits()),
                       framesPixmap.bytesPerLine());

    map_x = cv::Mat(img_source.size(), CV_32FC1);
    map_y = cv::Mat(img_source.size(), CV_32FC1);

    double a_off_projector = atan(190.0 / 1100.0);

    double projectionDistance = screenWidth / 2;
    // double pixel_per_mm = 1920 / screenWidth;
    double mm_per_pixel = screenWidth /1920.0;

    int frame_pixels = 400;
    

    double screenHeight = screenWidth * 1080 / 1920;

    double Aa = M_PI / 2 - a_off_projector;
    double B = tan(a_off_projector) * projectionDistance;
    double Ba = M_PI - Aa - mountAngle;
    double A = sin(Aa) / sin(Ba) * B;

    // top of image
    double y_conveyor = A;

    double C = y_conveyor * sin(mountAngle);
    double D = projectionDistance - C;

    double Ea = atan(y_conveyor / D);

    double y_virtual = projectionDistance * tan(Ea);

    // virtual top in mm
    double y_virt_top = y_virtual;

    for (int argi = 0; argi < map_x.rows; argi++)
    {
        for (int argj = 0; argj < map_x.cols; argj++)
        {

            double x = argj * mm_per_pixel + offsetX; // scale x

            double Aa = M_PI / 2 - a_off_projector;
            double B = tan(a_off_projector) * projectionDistance;
            double Ba = M_PI - Aa - mountAngle;
            double A = sin(Aa) / sin(Ba) * B;

            double y_conveyor = argi + A + offsetY;

            double C = y_conveyor * sin(mountAngle);
            double D = projectionDistance - C;

            double Ea = atan(y_conveyor / D);

            double y_virtual = projectionDistance * tan(Ea);

            double y = (y_virtual - y_virt_top) * mm_per_pixel;

            map_x.at<float>(argi, argj) = (float)x;
            map_y.at<float>(argi, argj) = (float)y;
            // std::cout << "map " << argj << " to " << y;
        }
    }
}

void Conveyor::calculateMapsPoly()
{

    cv::Mat img_source(framesPixmap.height(), framesPixmap.width(), CV_8UC4,
                       const_cast<uchar *>(framesPixmap.bits()),
                       framesPixmap.bytesPerLine());

    map_x = cv::Mat(img_source.size(), CV_32FC1);
    map_y = cv::Mat(img_source.size(), CV_32FC1);

    // double pixel_per_mm = 1920 / screenWidth;
    double mm_per_pixel = screenWidth /1920.0;

    int frame_pixels = 400;

    for (int argi = 0; argi < map_x.rows; argi++)
    {
        for (int argj = 0; argj < map_x.cols; argj++)
        {

            double y = y_factor(((argi + offsetY) * mm_per_pixel));
            double x = (argj + offsetX) * mm_per_pixel;

            map_x.at<float>(argi, argj) = (float)x;
            map_y.at<float>(argi, argj) = (float)y;
        }
    }
}

void Conveyor::drawDistorted()
{

    cv::Mat img_source(framesPixmap.height(), framesPixmap.width(), CV_8UC4,
                       const_cast<uchar *>(framesPixmap.bits()),
                       framesPixmap.bytesPerLine());

    cv::Mat img_warped(cv::Size(6000, 1080), CV_8UC4);

    cv::remap(img_source, img_warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    distoredPixmap = QImage(img_warped.data,
                            img_warped.cols, img_warped.rows,
                            static_cast<int>(img_warped.step),
                            QImage::Format_ARGB32)
                         .copy();
}

void Conveyor::purgeOld(int currentEncoderCount)
{

    int threshold_low = currentEncoderCount - 6000;
    int threshold_high = currentEncoderCount + 1000;

    Frames.erase(
        std::remove_if(
            Frames.begin(), Frames.end(),
            [&](Frame x)
            {
                if (x.getEncoderCount() < threshold_low)
                {
                    // the frame is off the end of the conveyor
                    return true;
                }
                else if (x.getEncoderCount() > threshold_high)
                {
                    // the frame is before the start of the conveyor
                    // maybe the encoder count rolled over.
                    return true;
                }

                return false;
            }),
        Frames.end());
}

void Conveyor::Add(Frame frame)
{
    purgeOld(frame.getEncoderCount());

    Frames.push_back(frame);

    // moved the creation of the thread higher up
    drawAll();

    // try
    // {
    //     drawThread = std::thread(&Conveyor::drawAll, this);

    //     drawThread.join();

    // }
    // catch (const std::exception &e)
    // {
    //     qInfo() << e.what();
    // }
}
