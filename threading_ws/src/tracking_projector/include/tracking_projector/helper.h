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

#ifndef HELPER_H
#define HELPER_H

#include <QBrush>
#include <QFont>
#include <QPen>
#include <QWidget>
#include <QPainterPath>
#include <math.h> /* pow */
#include "cherry_cpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "frame.h"
// #include <tf2/LinearMath/Quaternion.h>

//! [0]
class Helper
{
public:
    Helper();

public:
    void paint(QPainter *painter, QPaintEvent *event, std::vector<Cherry_cpp> cherries);
    void paint(QPainter *painter, QPaintEvent *event, QPixmap image, QPixmap internal, long mm_offset, long pixel_offset, long encoder_count);
    void paint_grid(QPainter *painter, QPaintEvent *event);
    QPainterPath lensDeform(
        const double screenWidth,
        const double mountAngle,
        const double offsetY);
    double circle_size = 50;
    double scaling = 891;
    double y_multiplier = 5 / 16 / 34; //(32.25 + (3.125-2.25)*2) / 32.25 ;
    // double y_factor(double y) { return (y * y * 0.000164746935628958) + (y * 0.84151250232418) - 0.920530171761975; }
    //  double y_factor(double y) { return (
    //      y * y * 0.00016444950208) +
    //      (y * 0.86208239700497) -
    //      10.8900679064569; }
    //  double y_factor(double y) { return (
    //      - y * y * 0.000162221678786572) +
    //      (y * 1.12813189636851)+
    //      14.4186812364243; }
    double y_factor(double y) { return (
                                           -y * y * 0.000176969104130805) +
                                       (y * 1.23068934149292) +
                                       15.7294704397464; }
    // double y_factor(double y) { return 1.0;}


private:
    QBrush background;
    QBrush circleBrush_ng;
    QBrush circleBrush_ok;
    QBrush circleBrush_side;
    QBrush circleBrush_maybe;
    QBrush gridBrush;
    QFont textFont;
    QPen circlePen;
    QPen textPen;
    QPen gridPen;
    QPainterPath distort;


    QPixmap conveyorScaled_ = QPixmap(3840,1080);

    // Frame frame;
};
//! [0]

#endif // HELPER_H
