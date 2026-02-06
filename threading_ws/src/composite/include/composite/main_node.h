#include <cstdio>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
// #include "rclcpp_components/register_node_macro.hpp"

#include "cherry_interfaces/srv/trigger.hpp"
#include "cherry_interfaces/action/find_cherries.hpp"
#include "cherry_interfaces/action/acquisitionhdr.hpp"
#include "cherry_interfaces/srv/trigger.hpp"
#include "cherry_interfaces/srv/combine_images.hpp"
#include "cherry_interfaces/msg/image_set.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cherry_interfaces/msg/image_set.hpp"
#include "cherry_interfaces/srv/detectionhdr.hpp"

using namespace std::chrono_literals;
  using FindCherries = cherry_interfaces::action::FindCherries;
  using Acquisition = cherry_interfaces::action::Acquisitionhdr;
  using GoalHandleAcquisition = rclcpp_action::ClientGoalHandle<Acquisition>;
  using GoalHandleFindCherries = rclcpp_action::ServerGoalHandle<FindCherries>;
  using ImageSet = cherry_interfaces::msg::ImageSet;
  using Image = sensor_msgs::msg::Image;
  using Detection = cherry_interfaces::srv::Detectionhdr;
  

class MainNode : public rclcpp::Node
{


public:
  MainNode();

private:
  // actoin server stuff
  // this will get the plc ready for capturing encoder counts
  // and start the image capture sequence aftwerwards
  rclcpp_action::Server<FindCherries>::SharedPtr detection_server_;
  rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID &uuid,
      std::shared_ptr<const FindCherries::Goal> goal);
  rclcpp_action::CancelResponse handle_cancel(
      const std::shared_ptr<GoalHandleFindCherries> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleFindCherries> goal_handle);
  void execute(const std::shared_ptr<GoalHandleFindCherries> goal_handle);
  std::future<void> find_cherries_future_;

  // connection stuff for starting an acqusition
  rclcpp_action::Client<Acquisition>::SharedPtr aquisition_client_;
  Acquisition::Result AcquireImage(int frame_id);

  // combine images cleint



  // plc eip stuff
  rclcpp::Client<cherry_interfaces::srv::Trigger>::SharedPtr plc_client_;
  bool ResetPLC(int frame_id);
  void acquire_goal_response_callback(const GoalHandleAcquisition::SharedPtr & goal_handle);
  void acquire_feedback_callback(
    GoalHandleAcquisition::SharedPtr,
    const std::shared_ptr<const Acquisition::Feedback> feedback);
  void acquire_result_callback(const GoalHandleAcquisition::WrappedResult & result);


  // combine images service
  rclcpp::Client<cherry_interfaces::srv::CombineImages>::SharedPtr image_combine_client_;
  // sensor_msgs::msg::Image CombineImages(Acquisition::Result images);


  //detection stuff

  rclcpp::Client<Detection>::SharedPtr detection_client_;
  Detection::Response::SharedPtr Detect(Acquisition::Result result, long frame_id);


};
