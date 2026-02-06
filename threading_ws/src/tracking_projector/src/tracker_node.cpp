#include "tracker_node.h"
#include <math.h>
#include <QDebug>
#include "rclcpp/rclcpp.hpp"
using namespace std::chrono_literals;
// using Alloc = std::pmr::polymorphic_allocator<void>;
// #include "./ui_tracker.h"

TrackerNode::TrackerNode() : rclcpp::Node("tracking_projector")
{

  subscription_points_ = this->create_subscription<cherry_interfaces::msg::CherryArray>(
      "/detection_server/detections", 1, std::bind(&TrackerNode::CherryArrayChanged, this, std::placeholders::_1));

  rclcpp::QoS encoderQoS = rclcpp::SensorDataQoS();
  rclcpp::Duration durr = rclcpp::Duration(0, 33333333); // 33ms deadline
  encoderQoS.deadline(durr);

  rclcpp::SubscriptionEventCallbacks event_callbacks = {.deadline_callback = std::bind(&TrackerNode::deadlineCallback, this, std::placeholders::_1)};
  // rclcpp::SubscriptionOptionsBase sub_options = {.event_callbacks = event_callbacks};

  rclcpp::SubscriptionOptions sub_opts;
  sub_opts.event_callbacks = event_callbacks;

  subscription_encoder_ = this->create_subscription<cherry_interfaces::msg::EncoderCount>(
      "encoder",
      encoderQoS,
      std::bind(&TrackerNode::EncoderCountChanged, this, std::placeholders::_1),
      sub_opts);

  // subscription_encoder_->set_on_new_qos_event_callback(&TrackerNode::deadlineCallback, RCL_SUBSCRIPTION_REQUESTED_DEADLINE_MISSED);

  // rotation_ = 0;
  rotation_matrix_ = {1.0, 0.0, 1.0, 0.0};

  this->declare_parameter<int>("x", conveyor.GetOffsetX());                       // mm
  this->declare_parameter<int>("y", conveyor.GetOffsetY());                       // mm
  this->declare_parameter<int>("screen_width", conveyor.GetScreenWidth()); //
  this->declare_parameter<double>("mount_angle", conveyor.GetMountAngle());
  this->declare_parameter<bool>("show_grid", show_grid_);
  this->declare_parameter<bool>("show_clean", conveyor.GetCleanVisibility());
  this->declare_parameter<bool>("show_pit", conveyor.GetPitVisibility());
  this->declare_parameter<bool>("show_maybe", conveyor.GetMaybeVisibility());
  this->declare_parameter<bool>("show_side", conveyor.GetSideVisibility());
  this->declare_parameter<int>("circle_size", Frame::GetCircleSize());
  // this->declare_parameter<double>("rotation", 0.0);
  this->declare_parameter<int>("screen", screen_);

  get_all_param();

  param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  cb_handle_x_ = param_subscriber_->add_parameter_callback("x", std::bind(&TrackerNode::cb_x_, this, std::placeholders::_1));
  cb_handle_y_ = param_subscriber_->add_parameter_callback("y", std::bind(&TrackerNode::cb_y_, this, std::placeholders::_1));
  cb_handle_scaling_ = param_subscriber_->add_parameter_callback("screen_width", std::bind(&TrackerNode::cb_screen_width_, this, std::placeholders::_1));
  cb_handle_showgrid_ = param_subscriber_->add_parameter_callback("show_grid", std::bind(&TrackerNode::cb_showgrid_, this, std::placeholders::_1));
  cb_handle_show_pit_ = param_subscriber_->add_parameter_callback("show_pit", std::bind(&TrackerNode::cb_show_pit_, this, std::placeholders::_1));
  cb_handle_show_clean_ = param_subscriber_->add_parameter_callback("show_clean", std::bind(&TrackerNode::cb_show_clean_, this, std::placeholders::_1));
  cb_handle_show_maybe_ = param_subscriber_->add_parameter_callback("show_maybe", std::bind(&TrackerNode::cb_show_maybe_, this, std::placeholders::_1));
  cb_handle_show_side_ = param_subscriber_->add_parameter_callback("show_side", std::bind(&TrackerNode::cb_show_side_, this, std::placeholders::_1));
  cb_handle_circlesize_ = param_subscriber_->add_parameter_callback("circle_size", std::bind(&TrackerNode::cb_circlesize_, this, std::placeholders::_1));
  // cb_handle_rotation_ = param_subscriber_->add_parameter_callback("rotation", std::bind(&TrackerNode::cb_rotation_, this, std::placeholders::_1));
  cb_handle_screen_ = param_subscriber_->add_parameter_callback("screen", std::bind(&TrackerNode::cb_screen_, this, std::placeholders::_1));
}

TrackerNode::~TrackerNode()
{
  delete update_timer_;
}

void TrackerNode::get_all_param()
{
    int x_, y_, screen_width_, rotation_, circle_size;
    double mount_angle_;
    bool show_grid_, show_clean_, show_pit_, show_side_, show_maybe_;

  this->get_parameter("x", x_);
  this->get_parameter("y", y_);
  this->get_parameter("screen_width", screen_width_);
  this->get_parameter("mount_angle", mount_angle_);
  this->get_parameter("show_grid", show_grid_);
  this->get_parameter("screen", screen_);
  this->get_parameter("show_clean", show_clean_);
  this->get_parameter("show_pit", show_pit_);
  this->get_parameter("show_maybe", show_maybe_);
  this->get_parameter("show_side", show_side_);
  this->get_parameter("circle_size", circle_size);
  // this->get_parameter("rotation", rotation_);
  rotation_matrix_ = {cos(rotation_), -sin(rotation_), sin(rotation_), cos(rotation_)};

  conveyor.SetCleanVisibility(show_clean_);
  conveyor.SetMaybeVisibility(show_maybe_);
  conveyor.SetPitVisibility(show_pit_);
  conveyor.SetSideVisibility(show_side_);
  conveyor.SetMountAngle(mount_angle_);
  conveyor.SetOffsetX(x_);
  conveyor.SetOffsetY(y_);
  conveyor.SetScreenWidth(screen_width_);
  //screen_cb_(screen_);
  Frame::SetCircleSize(circle_size);

}

int TrackerNode::GetScreen()
{
  return screen_;
}


void TrackerNode::CherryArrayChanged(cherry_interfaces::msg::CherryArray cherries)
{
  // update the internal array of cherries
  // i suppose I don;t need this
  // cherries_ = cherries;
  RCLCPP_DEBUG(this->get_logger(), "received frame, spnning up draw thread");

  // // translate the cherry values
  // std::vector<Cherry_cpp> cherries_translated = translate_msg(cherries);
  // Frame frame = Frame(cherries_translated, pixels(cherries.encoder_count));
  // conveyor.Add(frame);

  try
  {
    // RCLCPP_INFO(this->get_logger(), "Start draw thread");

    // if (drawFuture != NULL)
    // {
    //   auto status = drawFuture.wait_for(0s);
    //   if (status != std::future_status::ready)
    //   {
    //     RCLCPP_ERROR(this->get_logger(), "draw thread still running!");
    //     return;
    //   }
    // }

    drawFuture = std::async(&TrackerNode::addFrame, this, cherries);

    // drawThread.join();
    // RCLCPP_INFO(this->get_logger(), "join thread");
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "error starting draw thread: %s", e.what());
  }

  // int nextEncoderCount = cherries.encoder_count;
  // QImage nextPixMap = conveyor.getPixmap(encoderCount_);

  // // update the internal variable
  // // since this is also used by th animiate command, we use a mutex to prevent wierd
  // // behavior when both threads try to access the object at the same time.
  // cherry_mtx_.lock();
  // conveyorImage_ = nextPixMap;
  // referenceCount_ = nextEncoderCount;
  // cherry_mtx_.unlock();
}

void TrackerNode::addFrame(cherry_interfaces::msg::CherryArray cherries)
{
  // translate the cherry values
  try
  {
    std::vector<Cherry_cpp> cherries_translated = translate_msg(cherries);
    long mm = cherries.encoder_count; // encoder_to_mm(cherries.encoder_count);
    Frame frame = Frame(cherries_translated, mm);
    conveyor.Add(frame);

    // RCLCPP_INFO(this->get_logger(), "Encoder cb: '%ld'", encoderCount_);
    referenceMm_ = mm;
    frame_cb_(conveyor.getPixmapWarped(mm), conveyor.getPixmap(mm), mm);
    RCLCPP_INFO(this->get_logger(), "Added frame with  encoder mm: %ld", mm);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "error adding frame: %s", e.what());
  }
}

void TrackerNode::EncoderCountChanged(cherry_interfaces::msg::EncoderCount msg)
{

  // encoderCount_ = encoder_to_mm(msg.data);
  // hist.add(encoderCount_);
  // missed = 0;

  try
  {
    // RCLCPP_INFO(this->get_logger(), "Encoder cb: '%ld'", encoderCount_);
    encoder_cb_(msg.mm);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "error publishing encoder mm: %s", e.what());
  }
}

void TrackerNode::deadlineCallback(rclcpp::QOSDeadlineRequestedInfo info)
{

  // // esetimate what the encoder count should be
  // missed++;

  // if (missed > 15)
  //   return;

  // encoderCount_ = hist.predict(missed);

  RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 10000, "Encoder deadline missed");

  // try{
  //   RCLCPP_INFO(this->get_logger(), "Encoder cb: '%ld'", encoderCount_);
  //   encoder_cb_(encoderCount_);
  // } catch (const std::exception &e)
  // {
  //   RCLCPP_ERROR(this->get_logger(), "error publishing encoder count: %s", e.what());
  // }
}

std::vector<Cherry_cpp> TrackerNode::translate_msg(cherry_interfaces::msg::CherryArray msg)
{
  std::vector<Cherry_cpp> cherry_array = {};
  // cherry_interfaces::msg::CherryArray cherries_conveyor = msg.cherries;
  // RCLCPP_INFO(this->get_logger(), "Cherry vector length: '%ld'", cherries_conveyor.cherries.size());
  for (unsigned int argi = 0; argi < msg.cherries.size(); argi++)
  {
    Cherry_cpp cherry_conveyor(
        msg.cherries[argi].x * 1000,
        msg.cherries[argi].y * 1000,
        msg.cherries[argi].type);

    cherry_array.push_back(cherry_conveyor);

    // std::string s = std::string("cherry ") + cherry_conveyor.X);
    // qInfo() << "cherry " << cherry_conveyor.X << cherry_conveyor.Y << cherry_conveyor.Type;
  }

  return cherry_array;
}

void TrackerNode::cb_x_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter \"%s\" of type %s: \"%ld\"",
        p.get_name().c_str(),
        p.get_type_name().c_str(),
        p.as_int());
    conveyor.SetOffsetX(p.as_int());
    std::async(&TrackerNode::redraw, this);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'x': %s",
        e.what());
  }
}

void TrackerNode::cb_y_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter \"%s\" of type %s: \"%ld\"",
        p.get_name().c_str(),
        p.get_type_name().c_str(),
        p.as_int());
    conveyor.SetOffsetY(p.as_int());
    std::async(&TrackerNode::redraw, this);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'y': %s",
        e.what());
  }
}

void TrackerNode::cb_screen_width_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter \"%s\" of type %s: \"%ld\"",
        p.get_name().c_str(),
        p.get_type_name().c_str(),
        p.as_int());
    conveyor.SetScreenWidth(p.as_int());
    std::async(&TrackerNode::redraw, this);
    // helper->scaling = screen_width_;
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'screen_width': %s",
        e.what());
  }
}

void TrackerNode::cb_showgrid_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter 'show_grid'");
    show_grid_ = p.as_bool();
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'show_grid': %s",
        e.what());
  }
}

void TrackerNode::cb_show_clean_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter 'show_clean'");
    conveyor.SetCleanVisibility(p.as_bool());
    std::async(&TrackerNode::redraw, this);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'show_clean': %s",
        e.what());
  }
}

void TrackerNode::cb_show_pit_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter 'show_pit'");
    conveyor.SetPitVisibility(p.as_bool());
    std::async(&TrackerNode::redraw, this);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'show_pit': %s",
        e.what());
  }
}

void TrackerNode::cb_show_maybe_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter 'show_side'");
    conveyor.SetMaybeVisibility(p.as_bool());
    std::async(&TrackerNode::redraw, this);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'show_side': %s",
        e.what());
  }
}

void TrackerNode::cb_show_side_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter 'show_maybe'");
    conveyor.SetSideVisibility(p.as_bool());
    std::async(&TrackerNode::redraw, this);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'show_maybe': %s",
        e.what());
  }
}

void TrackerNode::cb_circlesize_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter \"%s\" of type %s: \"%f\"",
        p.get_name().c_str(),
        p.get_type_name().c_str(),
        p.as_double());
    Frame::SetCircleSize(p.as_int());
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'screen': %s",
        e.what());
  }
}

void TrackerNode::cb_screen_(const rclcpp::Parameter &p)
{
  try
  {
    RCLCPP_INFO(
        this->get_logger(), "cb: Received an update to parameter \"%s\" of type %s: \"%ld\"",
        p.get_name().c_str(),
        p.get_type_name().c_str(),
        p.as_int());
    screen_ = p.as_int();
    screen_cb_(screen_);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(
        this->get_logger(), "Error setting parameter 'show_grid': %s",
        e.what());
  }
}

// void TrackerNode::cb_rotation_(const rclcpp::Parameter &p)
// {
//   try
//   {
//     RCLCPP_INFO(
//         this->get_logger(), "cb: Received an update to parameter \"%s\" of type %s: \"%f\"",
//         p.get_name().c_str(),
//         p.get_type_name().c_str(),
//         p.as_double());

//     rotation_ = p.as_double();

//     // this is  rotation transformation using
//     // using [ x ] = [ cos(theta)  -sin(theta) ] * [ x' ]
//     //       [ y ]   [ sin(theta)   cos(theta) ]   [ y' ]
//     // we can get the point locations in a 180 deg
//     // rotated coordinate frame
//     // put into flattened vector to use later
//     rotation_matrix_ = {cos(rotation_), -sin(rotation_), sin(rotation_), cos(rotation_)};

//     RCLCPP_INFO(
//         this->get_logger(), "rotaition matrix values:  \n%f, %f, \n%f, %f",
//         rotation_matrix_[0],
//         rotation_matrix_[1],
//         rotation_matrix_[2],
//         rotation_matrix_[3]);
//   }
//   catch (const std::exception &e)
//   {
//     RCLCPP_ERROR(
//         this->get_logger(), "Error setting parameter 'rotation': %s",
//         e.what());
//   }
// }

void TrackerNode::SetFrameCallback(std::function<void(QImage projector, QImage conveyor, long reference_count)> func)
{
  frame_cb_ = func;
}

void TrackerNode::SetEncoderCallback(std::function<void(long number)> func)
{
  encoder_cb_ = func;
}

void TrackerNode::SetScreenCallback(std::function<void(int screen_number)> func)
{
  screen_cb_ = func;
}

void TrackerNode::redraw()
{
  conveyor.Redraw();
  frame_cb_(conveyor.getPixmapWarped(referenceMm_), conveyor.getPixmap(referenceMm_), referenceMm_);
  RCLCPP_INFO(this->get_logger(), "Redrew frame with  encoder mm: %ld", referenceMm_);
}