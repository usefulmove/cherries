/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "helper.h"

#include <QPainter>
#include <QPaintEvent>
#include <QWidget>
#include <QPaintEvent>

#include <rclcpp/rclcpp.hpp>

#include <QDebug>
#include <math.h>

//! [0]
Helper::Helper()
{
    QLinearGradient gradient(QPointF(50, -20), QPointF(80, 20));
    gradient.setColorAt(0.0, Qt::white);
    gradient.setColorAt(1.0, QColor(0xa6, 0xce, 0x39));

    background = QBrush(Qt::black);
    circleBrush_ok = QBrush(Qt::green);
    circleBrush_ng = QBrush(Qt::magenta);
    circleBrush_side = QBrush(Qt::cyan);
    circleBrush_maybe = QBrush(Qt::yellow);
    circlePen = QPen(Qt::black);
    circlePen.setWidth(1);
    textPen = QPen(Qt::white);
    textFont.setPixelSize(50);
    gridPen = QPen(Qt::green);
    gridPen.setWidth(1);
    gridBrush = QBrush(Qt::transparent);

    std::vector<Cherry_cpp> emptyList;
};

//! [0]
void Helper::paint(QPainter *painter, QPaintEvent *event, QPixmap image, QPixmap internal, long mm_offset, long pixel_offset, long encoder_count)
{
    RCLCPP_DEBUG(rclcpp::get_logger("helper"), "current: '%ld'", encoder_count);

    painter->fillRect(event->rect(), Qt::black);
    // QPainterPath distort = lensDeform(2262, 0.174, 0);

    QString encoder_str = QString("encoder_count: %1").arg(encoder_count);
    QString mm_offset_str = QString("mm_offset: %1").arg(mm_offset);
    QString pixel_offset_str = QString("pixel_offset: %1").arg(pixel_offset);

    int shift = 300;


    //double ratio = 0.5 ;//event->rect().width() / internal.width();
    // int offset_scaled = ratio * mm_offset;

    painter->drawPixmap(pixel_offset-shift, 0, image);
    //conveyorScaled_ = internal.scaled(internal.width() *ratio, internal.height() * ratio);
    //painter->drawPixmap(mm_offset*ratio, 1080, conveyorScaled_);

    painter->setPen(Qt::red);
    painter->setFont(QFont("Arial", 20));
    painter->drawText(QRect(10, 980, 500, 100), encoder_str);
    painter->drawText(QRect(500, 980, 500, 100), mm_offset_str);
    painter->drawText(QRect(1000, 980, 500, 100), pixel_offset_str);

    // if (paintInternal)
};

//! [1]
// void Helper::paint(QPainter *painter, QPaintEvent *event, std::vector<Cherry_cpp> cherries)
// {
//     // painter->translate(event->rect().bottomLeft());
//     // painter->scale(1.0,-1.0);
//     // transalte origin to bottom left, invert y axis

//     painter->fillRect(event->rect(), Qt::transparent );

//     // std::vector<Cherry_cpp> v;

//     // v.push_back( Cherry_cpp(0,0,2));
//     // v.push_back( Cherry_cpp(100,0,2));
//     // v.push_back( Cherry_cpp(0,100,2));
//     // v.push_back( Cherry_cpp(-100,0,2));
//     // v.push_back( Cherry_cpp(0,-100,2));
//     // v.push_back( Cherry_cpp(100,100,2));
//     // v.push_back( Cherry_cpp(-100,100,2));
//     // v.push_back( Cherry_cpp(-100,100,2));
//     // v.push_back( Cherry_cpp(100,-100,2));

//     Frame frame(cherries, 0);

//     //QPixmap img = QPixmap("/home/wesley/Pictures/Screenshots/Screenshot from 2023-05-05 15-27-53.png");

//     painter->drawPixmap(0,0, frame.getCleans());
//     painter->drawPixmap(0,0, frame.getSides());
//     painter->drawPixmap(0,0, frame.getMaybes());
//     painter->drawPixmap(0,0, frame.getPits());

//     // qInfo( "Drew frame" );
// }

// Deform scaled image to 1920x1080 screen
// do everything in mm and radian
QPainterPath Helper::lensDeform(
    const double screenWidth,
    const double mountAngle,
    const double offsetY)
{
    QPainterPath path;
    // path.addPath(source);

    // calculate the angles we need

    double a_off_projector = atan(190.0 / 1100.0);

    double projectionDistance = screenWidth / 2;
    double pixel_per_mm = 1920 / screenWidth;

    double screenHeight = screenWidth * 1080 / 1920;

    path.addRect(-screenWidth / 2, -screenHeight / 2, screenWidth, screenHeight);

    double Aa = M_PI - a_off_projector;
    double B = tan(a_off_projector) * projectionDistance;
    double Ba = M_PI - Aa - mountAngle;
    double A = sin(Aa) / sin(Ba) * B;

    // center of screen
    double y_conveyor = screenHeight + Aa;

    double C = y_conveyor * sin(mountAngle);
    double D = projectionDistance - C;

    double Ea = atan(y_conveyor / D);

    double y_virtual = projectionDistance * tan(Ea);

    // virtual zero in mm
    double y_virt_zero = y_virtual;

    for (int i = 0; i < path.elementCount(); ++i)
    {
        const QPainterPath::Element &e = path.elementAt(i);

        double x = e.x * pixel_per_mm; // scale x

        double Aa = M_PI - a_off_projector;
        double B = tan(a_off_projector) * projectionDistance;
        double Ba = M_PI - Aa - mountAngle;
        double A = sin(Aa) / sin(Ba) * B;

        // center of screen
        double y_conveyor = e.y + Aa;

        double C = y_conveyor * sin(mountAngle);
        double D = projectionDistance - C;

        double Ea = atan(y_conveyor / D);

        double y_virtual = projectionDistance * tan(Ea);

        double y = y_virtual * pixel_per_mm;

        path.setElementPositionAt(i, x, y);
    }

    return path;
}

//! [1]

//! [2]
void Helper::paint_grid(QPainter *painter, QPaintEvent *event)
{

    //     pel/in         pel/m     in/m
    double pixel_per_in = scaling / 39.370079;
    // draw a grid of 1 in squares

    painter->setBrush(gridBrush);
    gridPen.setColor(Qt::green);
    painter->setPen(gridPen);
    for (int argi = 0; argi < 1920 / pixel_per_in; argi++)
    {
        for (int argj = 0; argj < 1080 / pixel_per_in; argj++)
        {
            double y1 = y_factor((argj * pixel_per_in)) + 1;
            double y2 = y_factor(((argj + 1) * pixel_per_in)) + 1;
            int height = int(y2 - y1);
            painter->drawRect(
                argi * pixel_per_in + 1,
                y1,
                pixel_per_in,
                height);
        }
    }

    gridPen.setColor(Qt::red);
    painter->setPen(gridPen);
    double unit = (pixel_per_in * 12);
    for (int argi = 0; argi < 1920 / unit; argi++)
    {
        for (int argj = 0; argj < 1080 / unit; argj++)
        {

            double y1 = y_factor((argj * unit)) + 1;
            double y2 = y_factor(((argj + 1) * unit)) + 1;
            int height = int(y2 - y1);
            painter->drawRect(
                argi * unit + 1,
                y1,
                unit,
                height);
        }
    }
}

//! [2]
