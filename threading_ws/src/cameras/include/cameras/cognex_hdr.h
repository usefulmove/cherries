#ifndef COGNEX_CAM_CIC_5000_24_G_
#define COGNEX_CAM_CIC_5000_24_G_

#include <cstdio>
#include <future>
#include <thread>
#include <chrono>
#include <functional>
#include <condition_variable>
#include <mutex>
// #include "AcquisitionHelper.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include <VmbCPP/VmbCPP.h>
// #include "rclcpp_components/register_node_macro.hpp"

#include "cherry_interfaces/srv/reset_latches.hpp"
#include "cherry_interfaces/srv/encoder_latches.hpp"
#include "cherry_interfaces/srv/set_lights.hpp"
#include "cherry_interfaces/action/acquisitionhdr.hpp"
#include "cherry_interfaces/msg/encoder_count.hpp"
#include "cherry_interfaces/msg/inputs.hpp"
#include "cherry_interfaces/msg/image_set_hdr.hpp"
#include "cherry_interfaces/msg/trigger.hpp"

#include "std_msgs/msg/bool.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/fill_image.hpp"

#include "frameObserver.hpp"
#include "EventObserver.h"

using Acquisition = cherry_interfaces::action::Acquisitionhdr;
using GoalHandleAcquisition = rclcpp_action::ServerGoalHandle<Acquisition>;

class Cic5000CameraHdr : public rclcpp::Node
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
  rclcpp::Subscription<cherry_interfaces::msg::Trigger>::SharedPtr trigger_subscription_;

  // iamges
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image_top1_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image_top2_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image_top3_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image_bot1_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image_bot2_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_image_bot3_;
  rclcpp::Publisher<cherry_interfaces::msg::ImageSetHdr>::SharedPtr publisher_image_set_;

  VmbCPP::VmbSystem &m_vmbSystem;
  VmbCPP::CameraPtr m_camera;

  // frames to capture
  VmbCPP::FramePtr frame_bot1_;
  VmbCPP::FramePtr frame_bot2_;
  VmbCPP::FramePtr frame_bot3_;
  VmbCPP::FramePtr frame_top1_;
  VmbCPP::FramePtr frame_top2_;
  VmbCPP::FramePtr frame_top3_;
  std::condition_variable cv_frame_top1_;
  std::condition_variable cv_frame_top2_;
  std::condition_variable cv_frame_top3_;
  std::condition_variable cv_frame_bot1_;
  std::condition_variable cv_frame_bot2_;
  std::condition_variable cv_frame_bot3_;
  std::condition_variable cv_exposure_end_;
  std::condition_variable cv_trigger_ready_;
  std::condition_variable cv_trigger_notready_;

  VmbCPP::FeaturePtr eventFeature;

  double bot1_gain_ = 23.0;
  double bot2_gain_ = 23.0;
  double bot3_gain_ = 23.0;
  double top1_gain_ = 8.0;
  double top2_gain_ = 8.0;
  double top3_gain_ = 8.0;

  double bot1_exposure_ = 3000.0;
  double bot2_exposure_ = 1400.0;
  double bot3_exposure_ = 2000.0;
  double top1_exposure_ = 1350.0;
  double top2_exposure_ = 625.0;
  double top3_exposure_ = 2000.0;


  bool lastTriggerReady_;

public:
  Cic5000CameraHdr();
  ~Cic5000CameraHdr();

  void top_light_changed_callback(const std_msgs::msg::Bool &msg);
  void back_light_changed_callback(const std_msgs::msg::Bool &msg);

  void inputs_callback(const cherry_interfaces::msg::Inputs &msg);
  void encoder_callback(const cherry_interfaces::msg::EncoderCount &msg);

  void trigger_callback(const cherry_interfaces::msg::Trigger &msg);
  void execute_trigger(const cherry_interfaces::msg::Trigger &msg);

  void frameCallback_top1(const VmbCPP::FramePtr frame);
  void frameCallback_top2(const VmbCPP::FramePtr frame);
  void frameCallback_top3(const VmbCPP::FramePtr frame);
  void frameCallback_bot1(const VmbCPP::FramePtr frame);
  void frameCallback_bot2(const VmbCPP::FramePtr frame);
  void frameCallback_bot3(const VmbCPP::FramePtr frame);
  void stream_callback(const VmbCPP::FramePtr frame);

  void exposure_end_callback(const VmbCPP::FeaturePtr &feature);

  void CaptureSequence(uint64_t frame_id);

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

  int count_;
  int count_mm_;
  int stored_;
  int stored_mm_;

  std::shared_ptr<cherry_interfaces::action::Acquisitionhdr::Result> result_ = 
    std::make_shared<cherry_interfaces::action::Acquisitionhdr::Result>();

  cherry_interfaces::msg::ImageSetHdr image_set_ = 
    cherry_interfaces::msg::ImageSetHdr();

  // std::vector<cherry_interfaces::msg::ImageLayer> images_ = 
  //   std::vector<cherry_interfaces::msg::ImageLayer>();

  std::mutex trigger_mutex_;
  long frame_id_;

  // VmbErrorType SetGain(double value);

  std::string uint64_to_string( uint64_t value );

  void set_layer_data(const char* name, sensor_msgs::msg::Image image, uint64_t frame_id, int64_t count, int64_t mm);

};

#endif
