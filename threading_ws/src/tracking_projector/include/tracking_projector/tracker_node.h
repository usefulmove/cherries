#ifndef TRACKER_NODE_H
#define TRACKER_NODE_H

#include <QOpenGLWidget>
#include <QTimer>
#include <QPainter>
#include <QScreen>
#include <rclcpp/rclcpp.hpp>
// #include <tf2/LinearMath/Quaternion.h>
#include <math.h>
// #include "helper.h"
// #include "quaternion/quaternion.h"
#include <boost/math/quaternion.hpp>
#include <utility>
#include "std_msgs/msg/int64.hpp"
#include "cherry_interfaces/msg/cherry.hpp"
#include "cherry_interfaces/msg/cherry_array.hpp"
#include "cherry_interfaces/msg/cherry_array_stamped.hpp"
#include "cherry_interfaces/msg/encoder_count.hpp"
#include <mutex> // std::mutex
// header
#include <vector>
#include "cherry_cpp.hpp"
#include "conveyor.h"
#include "history.h"
#include <functional>

// QT_BEGIN_NAMESPACE
// namespace Ui { class tracker; }
// QT_END_NAMESPACE

class TrackerNode : public rclcpp::Node
{

public:
    // Tracker();
    TrackerNode();
    ~TrackerNode();

    void SetFrameCallback(std::function<void(QImage conveyor, QImage projector, long reference_count)> func);
    void SetEncoderCallback(std::function<void(long number)> func);
    void SetScreenCallback(std::function<void(int screen_number)> func);
    void SetShowGridCallback(std::function<void(bool show_grid, QImage grid)> func);

    int GetScreen();



private:
    // Helper *helper;
    int elapsed;
    QTimer *update_timer_;

    rclcpp::Subscription<cherry_interfaces::msg::CherryArray>::SharedPtr subscription_points_;
    rclcpp::Subscription<cherry_interfaces::msg::EncoderCount>::SharedPtr subscription_encoder_;
    void CherryArrayChanged(cherry_interfaces::msg::CherryArray msg);
    void EncoderCountChanged(cherry_interfaces::msg::EncoderCount msg);
    cherry_interfaces::msg::CherryArrayStamped cherries_;
    std::vector<Cherry_cpp> cherries_translated_;

    void get_all_param();

    // X,Y, theta location of the projector
    std::vector<double> rotation_matrix_ = {1.0, 0.0, 1.0, 0.0};
    int screen_ = 1;
    bool show_grid_;



    std::vector<Cherry_cpp> translate_msg(cherry_interfaces::msg::CherryArray cherries);

    std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
    void cb_x_(const rclcpp::Parameter &p);
    void cb_y_(const rclcpp::Parameter &p);
    void cb_screen_width_(const rclcpp::Parameter &p);
    void cb_showgrid_(const rclcpp::Parameter &p);
    void cb_show_clean_(const rclcpp::Parameter &p);
    void cb_show_pit_(const rclcpp::Parameter &p);
    void cb_show_maybe_(const rclcpp::Parameter &p);
    void cb_show_side_(const rclcpp::Parameter &p);
    void cb_circlesize_(const rclcpp::Parameter &p);
    // void cb_rotation_(const rclcpp::Parameter &p);
    void cb_screen_(const rclcpp::Parameter &p);

    void redraw();

    Conveyor conveyor = Conveyor();
    QImage conveyorImage_ = QImage(6000, 1080, QImage::Format_ARGB32_Premultiplied);
    QImage warpedImage_ = QImage(6000, 1080, QImage::Format_ARGB32_Premultiplied);
    long referenceMm_ = 0;
    long encoderCount_ = 0;

    int missed = 0;

    // QPixmap warped(2920, 1080);
    // QPixmap conveyor(4840, 1080);
    long reference_count;

    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_x_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_y_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_scaling_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_showgrid_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_show_pit_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_show_clean_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_show_maybe_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_show_side_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_circlesize_;
    // std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_rotation_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_screen_;
    // std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_show_pit_;

    // use this to handle unreliebale encoder data
    History hist;
    void deadlineCallback(rclcpp::QOSDeadlineRequestedInfo info);

    int encoder_to_mm(int encoderCount) { return int(encoderCount * 0.319185813604723); };

    std::future<void> drawFuture;
    void addFrame(cherry_interfaces::msg::CherryArray cherries);

    std::function<void(long number)> encoder_cb_;
    std::function<void(QImage projector, QImage conveyor, long reference_count)> frame_cb_;
    std::function<void(int screen_number)> screen_cb_;
    std::function<void(bool show_grid, QImage grid)> grid_cb_;
};

#endif // TRACKER_H