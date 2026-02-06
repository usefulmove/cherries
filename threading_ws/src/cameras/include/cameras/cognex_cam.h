#ifndef COGNEX_CAM_CIC_5000_24_G_
#define COGNEX_CAM_CIC_5000_24_G_

#include <cstdio>
#include <future>
#include <thread>
#include <chrono>
#include <functional>
#include <condition_variable>
// #include "AcquisitionHelper.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include <VmbCPP/VmbCPP.h>
// #include "rclcpp_components/register_node_macro.hpp"

#include "cherry_interfaces/srv/reset_latches.hpp"
#include "cherry_interfaces/srv/encoder_latches.hpp"
#include "cherry_interfaces/srv/set_lights.hpp"
#include "cherry_interfaces/action/acquisition.hpp"
#include "cherry_interfaces/msg/encoder_count.hpp"
#include "cherry_interfaces/msg/inputs.hpp"

#include "std_msgs/msg/bool.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/fill_image.hpp"

#include "frameObserver.hpp"
#include "EventObserver.h"

using Acquisition = cherry_interfaces::action::Acquisition;
using GoalHandleAcquisition = rclcpp_action::ServerGoalHandle<Acquisition>;

class Cic5000Camera : public rclcpp::Node
{

private:
  std::future<void> future_;
  std::thread capture_thread_;

  bool acquiring_;
  bool top_light_status_;
  bool back_light_status_;
  rclcpp_action::Server<Acquisition>::SharedPtr acquisition_server_;
  //  AcquisitionHelper acquisitionHelper;
  rclcpp::Client<cherry_interfaces::srv::ResetLatches>::SharedPtr plc_client_reset_latches_;
  // rclcpp::Client<cherry_interfaces::srv::EncoderLatches> plc_client_get_latches_;
  rclcpp::Client<cherry_interfaces::srv::SetLights>::SharedPtr plc_client_set_lights_;
  rclcpp::Subscription<cherry_interfaces::msg::EncoderCount>::SharedPtr encoder_subscription_;
  rclcpp::Subscription<cherry_interfaces::msg::Inputs>::SharedPtr plc_inputs_subscription_;

  // iamges
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image_top_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image_bot_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_images_;

  VmbCPP::VmbSystem &m_vmbSystem;
  VmbCPP::CameraPtr m_camera;

  // frames to capture
  VmbCPP::FramePtr frame_bot_;
  VmbCPP::FramePtr frame_top_;
  std::condition_variable cv_frame_top_;
  std::condition_variable cv_frame_bot_;
  std::condition_variable cv_exposure_end_;
  std::condition_variable cv_trigger_ready_;
  std::condition_variable cv_trigger_notready_;

  VmbCPP::FeaturePtr eventFeature;

  double bot_gain_ = 20.0;
  double top_gain_ = 5.0;

  bool lastTriggerReady_;

public:
  Cic5000Camera();
  ~Cic5000Camera();

  void top_light_changed_callback(const std_msgs::msg::Bool &msg);
  void back_light_changed_callback(const std_msgs::msg::Bool &msg);

  void inputs_callback(const cherry_interfaces::msg::Inputs &msg);
  void encoder_callback(const cherry_interfaces::msg::EncoderCount &msg);

  void frameCallback_top(const VmbCPP::FramePtr frame);
  void frameCallback_bot(const VmbCPP::FramePtr frame);
  void stream_callback(const VmbCPP::FramePtr frame);

  void exposure_end_callback(const VmbCPP::FeaturePtr &feature);

  void CaptureSequence();

  /**
   * \brief The constructor will initialize the API and open the given camera
   *
   * \param[in] pCameraId  zero terminated C string with the camera id for the camera to be used
   */
  void InitializeCamera(const char *cameraId);

  /**
   * \brief Start the acquisition.
   */
  void Start();

  /**
   * \brief Stop the acquisition.
   */
  void Stop();

  // action server stuff
  // handle a goal request
  rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID &uuid,
      std::shared_ptr<const Acquisition::Goal> goal);

  // cancel an exisitng goal
  rclcpp_action::CancelResponse handle_cancel(
      const std::shared_ptr<GoalHandleAcquisition> goal_handle);

  void handle_accepted(const std::shared_ptr<GoalHandleAcquisition> goal_handle);

  void execute(const std::shared_ptr<GoalHandleAcquisition> goal_handle);

private:
  void RegisterEvents();
  VmbErrorType ActivateNotification();

  int start_top_;
  int start_top_mm_;
  int start_bot_;
  int start_bot_mm_;

  int count_;
  int count_mm_;
  int stored_;
  int stored_mm_;

  std::shared_ptr<cherry_interfaces::action::Acquisition::Result> result_ = 
    std::make_shared<cherry_interfaces::action::Acquisition::Result>();




  long frame_id_;

  // VmbErrorType SetGain(double value);
};

#endif
