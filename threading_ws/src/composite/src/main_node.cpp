#include <cstdio>
#include "main_node.h"
#include <future>


MainNode::MainNode()
    : Node("main")
{

  detection_server_ = rclcpp_action::create_server<FindCherries>(
      this,
      "find_cherries",
      std::bind(&MainNode::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&MainNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&MainNode::handle_accepted, this, std::placeholders::_1));

  plc_client_ = this->create_client<cherry_interfaces::srv::Trigger>("start_capture");

  aquisition_client_ = rclcpp_action::create_client<Acquisition>(
      this,
      "acquisition");

  image_combine_client_ = this->create_client<cherry_interfaces::srv::CombineImages>("image_combiner/combine");

  detection_client_ = this->create_client<Detection>("detection_server/detect");
};

rclcpp_action::GoalResponse MainNode::handle_goal(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const FindCherries::Goal> goal)
{

  RCLCPP_INFO(this->get_logger(), "accept find cherries");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
};

rclcpp_action::CancelResponse MainNode::handle_cancel(
    const std::shared_ptr<GoalHandleFindCherries> goal_handle)
{

  RCLCPP_INFO(this->get_logger(), "Received request to cancel find cherries");
  (void)goal_handle;

  // TO DO kill the acquisition thread if running
  return rclcpp_action::CancelResponse::ACCEPT;
};

void MainNode::handle_accepted(const std::shared_ptr<GoalHandleFindCherries> goal_handle)
{
  RCLCPP_INFO(this->get_logger(), "Starting find cherries thread");
  find_cherries_future_ = std::async(std::launch::async, &MainNode::execute, this, goal_handle);
};

void MainNode::execute(const std::shared_ptr<GoalHandleFindCherries> goal_handle)
{
  try
  {

    const auto goal = goal_handle->get_goal();
    // ResetPLC(goal->frame_id);
    auto acquisition_result = AcquireImage(goal->frame_id);
    auto detection_result = Detect(acquisition_result, acquisition_result.mm_bot1);

    auto result = std::make_shared<cherry_interfaces::action::FindCherries::Result>();
    result->frame_id = goal->frame_id;
    result->cherries = detection_result->cherries;
    result->status = 0;
    RCLCPP_INFO(this->get_logger(), "goal frame id: %ld", goal->frame_id);

    goal_handle->succeed(result);
  }
  catch (std::exception &e)
  {
    RCLCPP_INFO(this->get_logger(), "Error finding cherries: %s", e.what());
  }
};

Acquisition::Result MainNode::AcquireImage(int frame_id)

{

  auto goal_msg = Acquisition::Goal();
  goal_msg.frame_id = frame_id;

  auto send_goal_options = rclcpp_action::Client<Acquisition>::SendGoalOptions();
  send_goal_options.goal_response_callback =
      std::bind(&MainNode::acquire_goal_response_callback, this, std::placeholders::_1);
  send_goal_options.feedback_callback =
      std::bind(&MainNode::acquire_feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
  send_goal_options.result_callback =
      std::bind(&MainNode::acquire_result_callback, this, std::placeholders::_1);
  auto acquisition_future = aquisition_client_->async_send_goal(goal_msg, send_goal_options);
  // auto acquisition_future = aquisition_client_->async_send_goal(goal_msg);
  std::future_status status = acquisition_future.wait_for(1s);

  if (status == std::future_status::ready)
  {
    auto result = aquisition_client_->async_get_result(acquisition_future.get());
    auto thing2 = result.get();
    auto thing3 = thing2.result;
    auto thing4 = thing3->image_bot1;
    return *thing3;
  }

  throw std::runtime_error("Timeout acquiring images");
};

void MainNode::acquire_goal_response_callback(const GoalHandleAcquisition::SharedPtr &goal_handle){};
void MainNode::acquire_feedback_callback(
    GoalHandleAcquisition::SharedPtr,
    const std::shared_ptr<const Acquisition::Feedback> feedback){};
void MainNode::acquire_result_callback(const GoalHandleAcquisition::WrappedResult &result){};

bool MainNode::ResetPLC(int frame_id)
{

  auto request = std::make_shared<cherry_interfaces::srv::Trigger::Request>();
  request->id = frame_id;

  auto future_plc_reset = plc_client_->async_send_request(request);

  std::future_status status = future_plc_reset.wait_for(1s);
  if (status == std::future_status::ready)
    return true;

  throw std::runtime_error("Timeout reseting plc");
};

// sensor_msgs::msg::Image MainNode::CombineImages(Acquisition::Result images)
// {

//   auto img_set = std::make_shared<cherry_interfaces::msg::ImageSet>();
//   img_set->frame_id = images.frame_id;
//   img_set->image_top = images.image_top;
//   img_set->image_bot = images.image_bot;
//   img_set->count_top = images.count_top;
//   img_set->count_bot = images.count_bot;
//   img_set->mm_top = images.mm_top;
//   img_set->mm_bot = images.mm_bot;

//   auto request = std::make_shared<cherry_interfaces::srv::CombineImages::Request>();
//   request->image_set = *img_set;

//   auto future_combine_images = image_combine_client_->async_send_request(request);

//   std::future_status status = future_combine_images.wait_for(1s);
//   if (status == std::future_status::ready)
//     return future_combine_images.get()->image;

//   throw std::runtime_error("Timeout combining images");
// };

Detection::Response::SharedPtr MainNode::Detect(Acquisition::Result result, long frame_id)
{
  auto request =
      std::make_shared<Detection::Request>();

  // request->image_top1 = result.image_top1;
  request->image_top2 = result.image_top2;
  request->image_bot1 = result.image_bot1;
  request->image_bot2 = result.image_bot2;

  request->count_top1 = result.count_top1;
  request->count_top2 = result.count_top2;
  request->count_bot1 = result.count_bot1;
  request->count_bot2 = result.count_bot2;

  request->mm_top1 = result.mm_top1;
  request->mm_top2 = result.mm_top2;
  request->mm_bot1 = result.mm_bot1;
  request->mm_bot2 = result.mm_bot2;

  request->frame_id = result.frame_id;

  auto detection_future = detection_client_->async_send_request(request);

  auto status = detection_future.wait_for(10s); // will be slow using only cpu

  if (status == std::future_status::ready)
    return detection_future.get();
};

bool spin = true;
void my_handler(int s)
{
  (void)s;
  spin = false;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto main_node = std::make_shared<MainNode>();

  rclcpp::spin(main_node);

  rclcpp::shutdown();
  return 0;
}
