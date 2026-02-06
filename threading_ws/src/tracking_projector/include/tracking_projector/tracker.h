#ifndef TRACKER_H
#define TRACKER_H

#include <QOpenGLWidget>
#include <QTimer>
#include <QPainter>
#include <QScreen>
#include <rclcpp/rclcpp.hpp>
//#include <tf2/LinearMath/Quaternion.h>
#include <math.h>
#include "helper.h"
//#include "quaternion/quaternion.h"
#include <boost/math/quaternion.hpp>
#include <utility>
#include "std_msgs/msg/int64.hpp"
#include "cherry_interfaces/msg/cherry.hpp"
#include "cherry_interfaces/msg/cherry_array.hpp"
#include "cherry_interfaces/msg/cherry_array_stamped.hpp"
#include <mutex>          // std::mutex
#include <future>
// header
#include <vector>
#include "cherry_cpp.hpp"
#include "conveyor.h"
#include "history.h"
#include <signal.h>


// QT_BEGIN_NAMESPACE
// namespace Ui { class tracker; }
// QT_END_NAMESPACE




class Tracker : public QOpenGLWidget
{
    Q_OBJECT

public:
    //Tracker();
    Tracker(Helper *helper, std::function<void (int number)> Screen_cb, QWidget *parent = nullptr);
    ~Tracker();

    void SetEncoder(long count);
    void SetImage(QImage projector, QImage conveyor, long reference_count);
    void SetScreen(QScreen *screen_number);


void animate();

private slots:
  void onUpdate();

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    Helper *helper;
    int elapsed;
    QTimer* update_timer_;
    rclcpp::Node::SharedPtr nh_;
    
    void CherryArrayChanged(cherry_interfaces::msg::CherryArray msg);
    void EncoderCountChanged(std_msgs::msg::Int64 msg);
    cherry_interfaces::msg::CherryArrayStamped cherries_;
    std::vector<Cherry_cpp> cherries_translated_;
    std::mutex cherry_mtx_;           // mutex for critical section
    

        // X,Y, theta location of the projector
    double x_, y_, scaling_factor_, rotation_;
    bool show_grid_;
    std::vector<double> rotation_matrix_ = { 1.0, 0.0, 1.0, 0.0 };
    QScreen *screen_;
    int screen_number_;
    bool change_screen_ = false;

    std::function<void (int number)> screen_changed_callback_;

    Cherry_cpp translate(Cherry_cpp cherry);
    std::vector<Cherry_cpp> translate_msg(cherry_interfaces::msg::CherryArray cherries);

    
    std::mutex image_mtx_;           // mutex for critical section
    Conveyor conveyor = Conveyor();
    QPixmap conveyorImage_ = QPixmap(6000,1080);

    QPixmap projectorImage_ = QPixmap(6000,1080);
    long referenceCount_ = 0;
    long referenceMm_ = 0;
    long referencePixels_ = 0;

    long encoderCount_ = 0;
    long mm_ = 0;
    long pixels_ = 0;
    
    int missed = 0;


    // std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_show_pit_;

    //use this to handle unreliebale encoder data
    History hist;
    void deadlineCallback(rclcpp::QOSDeadlineRequestedInfo info);

    long encoder_to_mm(long encoderCount) {return long(encoderCount * 0.319185813604723); };
    int screen_width = 2154;
    long mm_to_pixels(long mm) { return long( mm * 1920.0 / 2154  ); }

    void addFrame(cherry_interfaces::msg::CherryArray cherries);

    
    void set_screen();
        // struct sigaction sigIntHandler;


};

// void myHandlerInTracker (int signum){
//   Tracker::spin_ = false;
// }

#endif // TRACKER_H
