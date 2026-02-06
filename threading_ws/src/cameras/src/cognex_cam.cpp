#include "cognex_cam.h"

using namespace std::chrono_literals;

using Acquisition = cherry_interfaces::action::Acquisition;
using GoalHandleAcquisition = rclcpp_action::ServerGoalHandle<Acquisition>;

// AcquisitionHelper::AcquisitionHelper(std::function<void(std::string)> func) :
//     AcquisitionHelper(nullptr)
// {
//     logger = func;
// }

/**
 * \brief Helper function to adjust the packet size for Allied vision GigE cameras
 */
void GigEAdjustPacketSize(VmbCPP::CameraPtr camera)
{
  VmbCPP::StreamPtrVector streams;
  VmbErrorType err = camera->GetStreams(streams);

  if (err != VmbErrorSuccess || streams.empty())
  {
    throw std::runtime_error("Could not get stream modules, err=" + std::to_string(err));
  }

  VmbCPP::FeaturePtr feature;
  err = streams[0]->GetFeatureByName("GVSPAdjustPacketSize", feature);

  if (err == VmbErrorSuccess)
  {
    err = feature->RunCommand();
    if (err == VmbErrorSuccess)
    {
      bool commandDone = false;
      do
      {
        if (feature->IsCommandDone(commandDone) != VmbErrorSuccess)
        {
          break;
        }
      } while (commandDone == false);
    }
    else
    {
      std::cout << "Error while executing GVSPAdjustPacketSize, err=" + std::to_string(err) << std::endl;
    }
  }
}

// put the image buffer from a frame into a sensor::image message
bool frameToImage(const VmbCPP::FramePtr vimba_frame_ptr, sensor_msgs::msg::Image &image)
{
  VmbPixelFormatType pixel_format;
  VmbUint32_t width, height, nSize;

  vimba_frame_ptr->GetWidth(width);
  vimba_frame_ptr->GetHeight(height);
  vimba_frame_ptr->GetPixelFormat(pixel_format);
  vimba_frame_ptr->GetBufferSize(nSize);

  VmbUint32_t step = nSize / height;

  // NOTE: YUV and ARGB formats not supported
  std::string encoding;
  if (pixel_format == VmbPixelFormatMono8)
    encoding = sensor_msgs::image_encodings::MONO8;
  else if (pixel_format == VmbPixelFormatMono10)
    encoding = sensor_msgs::image_encodings::MONO16;
  else if (pixel_format == VmbPixelFormatMono12)
    encoding = sensor_msgs::image_encodings::MONO16;
  else if (pixel_format == VmbPixelFormatMono12Packed)
    encoding = sensor_msgs::image_encodings::MONO16;
  else if (pixel_format == VmbPixelFormatMono14)
    encoding = sensor_msgs::image_encodings::MONO16;
  else if (pixel_format == VmbPixelFormatMono16)
    encoding = sensor_msgs::image_encodings::MONO16;
  else if (pixel_format == VmbPixelFormatBayerGR8)
    encoding = sensor_msgs::image_encodings::BAYER_GRBG8;
  else if (pixel_format == VmbPixelFormatBayerRG8)
    encoding = sensor_msgs::image_encodings::BAYER_RGGB8;
  else if (pixel_format == VmbPixelFormatBayerGB8)
    encoding = sensor_msgs::image_encodings::BAYER_GBRG8;
  else if (pixel_format == VmbPixelFormatBayerBG8)
    encoding = sensor_msgs::image_encodings::BAYER_BGGR8;
  else if (pixel_format == VmbPixelFormatBayerGR10)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerRG10)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerGB10)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerBG10)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerGR12)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerRG12)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerGB12)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerBG12)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerGR12Packed)
    encoding = sensor_msgs::image_encodings::TYPE_32SC4;
  else if (pixel_format == VmbPixelFormatBayerRG12Packed)
    encoding = sensor_msgs::image_encodings::TYPE_32SC4;
  else if (pixel_format == VmbPixelFormatBayerGB12Packed)
    encoding = sensor_msgs::image_encodings::TYPE_32SC4;
  else if (pixel_format == VmbPixelFormatBayerBG12Packed)
    encoding = sensor_msgs::image_encodings::TYPE_32SC4;
  else if (pixel_format == VmbPixelFormatBayerGR16)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerRG16)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerGB16)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatBayerBG16)
    encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  else if (pixel_format == VmbPixelFormatRgb8)
    encoding = sensor_msgs::image_encodings::RGB8;
  else if (pixel_format == VmbPixelFormatBgr8)
    encoding = sensor_msgs::image_encodings::BGR8;
  else if (pixel_format == VmbPixelFormatRgba8)
    encoding = sensor_msgs::image_encodings::RGBA8;
  else if (pixel_format == VmbPixelFormatBgra8)
    encoding = sensor_msgs::image_encodings::BGRA8;
  else if (pixel_format == VmbPixelFormatRgb12)
    encoding = sensor_msgs::image_encodings::TYPE_16UC3;
  else if (pixel_format == VmbPixelFormatRgb16)
    encoding = sensor_msgs::image_encodings::TYPE_16UC3;
  else
    // RCLCPP_WARN(logger_, "Received frame with unsupported pixel format %d", pixel_format);
    if (encoding == "")
      return false;

  VmbUchar_t *buffer_ptr;
  VmbErrorType err = vimba_frame_ptr->GetImage(buffer_ptr);
  bool res = false;
  if (VmbErrorSuccess == err)
  {
    res = sensor_msgs::fillImage(image, encoding, height, width, step, buffer_ptr);
  }
  else
  {
    // RCLCPP_ERROR_STREAM(logger_, "Could not GetImage. "
    //                                   << "\n Error: " << errorCodeToMessage(err));
  }
  return res;
}

Cic5000Camera::Cic5000Camera() : Node("cam_cic_5000_24_g"),
                                 m_vmbSystem(VmbCPP::VmbSystem::GetInstance())
{

  InitializeCamera(nullptr);

  // CaptureSequence();

  this->acquisition_server_ = rclcpp_action::create_server<Acquisition>(
      this,
      "acquisition",
      std::bind(&Cic5000Camera::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Cic5000Camera::handle_cancel, this, std::placeholders::_1),
      std::bind(&Cic5000Camera::handle_accepted, this, std::placeholders::_1));

  plc_client_reset_latches_ = this->create_client<cherry_interfaces::srv::ResetLatches>("reset_latches");
  // plc_client_get_latches_ = this->create_client<cherry_interfaces::srv::GetLatches>("get_latches");
  plc_client_set_lights_ = this->create_client<cherry_interfaces::srv::SetLights>("set_lights");

  rclcpp::QoS publisherQoS = rclcpp::SensorDataQoS();
  publisherQoS.deadline(1000us);
  publisherQoS.lifespan(1000us);

  encoder_subscription_ = this->create_subscription<cherry_interfaces::msg::EncoderCount>(
      "encoder",
      publisherQoS,
      std::bind(
          &Cic5000Camera::encoder_callback,
          this,
          std::placeholders::_1));
  plc_inputs_subscription_ = this->create_subscription<cherry_interfaces::msg::Inputs>(
      "inputs",
      publisherQoS,
      std::bind(
          &Cic5000Camera::inputs_callback,
          this,
          std::placeholders::_1));

  publisher_image_top_ = this->create_publisher<sensor_msgs::msg::Image>("top_image", 10);
  publisher_image_bot_ = this->create_publisher<sensor_msgs::msg::Image>("bot_image", 10);
  publisher_images_ = this->create_publisher<sensor_msgs::msg::Image>("images", 10);
}

Cic5000Camera::~Cic5000Camera()
{

  try
  {
    VmbCPP::FeaturePtr pFeature;
    m_camera->GetFeatureByName("AcquisitionEnd", pFeature);
    pFeature->RunCommand();

    m_camera->EndCapture();
    // Stop();
  }
  catch (std::runtime_error &e)
  {
    std::cout << e.what() << std::endl;
  }
  catch (...)
  {
    // ignore
  }

  m_vmbSystem.Shutdown();
}

void Cic5000Camera::inputs_callback(const cherry_interfaces::msg::Inputs &msg)
{
  bool triggerReady = msg.trigger_ready;

  if (lastTriggerReady_ != triggerReady)
  {
    RCLCPP_ERROR(this->get_logger(), "trigger ready = %d", triggerReady);
    lastTriggerReady_ = triggerReady;
  }

  if (triggerReady)
  {
    cv_trigger_ready_.notify_one();
  }
  else
  {

    cv_trigger_notready_.notify_one();
  }
}
void Cic5000Camera::encoder_callback(const cherry_interfaces::msg::EncoderCount &msg)
{
  count_ = msg.count;
  count_mm_ = msg.mm;
  stored_ = msg.count_stored;
  stored_mm_ = msg.mm_stored;
}

void Cic5000Camera::top_light_changed_callback(const std_msgs::msg::Bool &msg)
{
  top_light_status_ = msg.data;
}

void Cic5000Camera::back_light_changed_callback(const std_msgs::msg::Bool &msg)
{
  back_light_status_ = msg.data;
}

// action server stuff
// handle a goal request
rclcpp_action::GoalResponse Cic5000Camera::handle_goal(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const Acquisition::Goal> goal)
{
  RCLCPP_INFO(this->get_logger(), "Received aquisition with frame id: %ld", goal->frame_id);
  (void)uuid; // acknowledge tha the uuid is not used so we do not get a warning.

  frame_id_ = goal->frame_id;

  // TODO check if a acquisition is running
  // if (future_ ){
  //   auto status = future_.wait_for(0ms);

  //   // Print status.
  //   if (status != std::future_status::ready) {
  //       return rclcpp_action::GoalResponse::REJECT;
  //   }
  // }
  // frame_id_ = std::to_string(frame_id_uint_);
  RCLCPP_INFO(this->get_logger(), "accepting goal");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

// cancel an exisitng goal
rclcpp_action::CancelResponse Cic5000Camera::handle_cancel(
    const std::shared_ptr<GoalHandleAcquisition> goal_handle)
{
  RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
  (void)goal_handle;

  // TO DO kill the acquisition thread if running
  return rclcpp_action::CancelResponse::ACCEPT;
}

void Cic5000Camera::handle_accepted(const std::shared_ptr<GoalHandleAcquisition> goal_handle)
{
  using namespace std::placeholders;
  // this needs to return quickly to avoid blocking the executor, so spin up a new thread
  RCLCPP_INFO(this->get_logger(), "Starting acquisition thread");
  future_ = std::async(std::launch::async, &Cic5000Camera::execute, this, goal_handle);
}

void Cic5000Camera::execute(const std::shared_ptr<GoalHandleAcquisition> goal_handle)
{

  try
  {
    result_ = std::make_shared<cherry_interfaces::action::Acquisition::Result>();
    CaptureSequence();

    // sensor_msgs::msg::Image top_img_msg = sensor_msgs::msg::Image();
    // sensor_msgs::msg::Image bot_img_msg = sensor_msgs::msg::Image();
    // frameToImage(frame_top_, top_img_msg);
    // frameToImage(frame_bot_, bot_img_msg);

    result_->frame_id = frame_id_;

    goal_handle->succeed(result_);
    RCLCPP_INFO(this->get_logger(), "Handled goal succesfully");
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "Error setting goal message, err='%s'", e.what());
  }
}

void Cic5000Camera::InitializeCamera(const char *cameraId)
{
  VmbErrorType err = m_vmbSystem.Startup();

  if (err != VmbErrorSuccess)
  {
    RCLCPP_ERROR(this->get_logger(), "Could not start API, err='%s'", std::to_string(err).data());
    throw std::runtime_error("Could not start API, err=" + std::to_string(err));
  }

  VmbCPP::CameraPtrVector cameras;
  err = m_vmbSystem.GetCameras(cameras);
  if (err != VmbErrorSuccess)
  {
    m_vmbSystem.Shutdown();
    RCLCPP_ERROR(this->get_logger(), "Could not get cameras, eerr='%s'", std::to_string(err).data());
    throw std::runtime_error("Could not get cameras, err=" + std::to_string(err));
  }

  if (cameras.empty())
  {
    m_vmbSystem.Shutdown();
    RCLCPP_ERROR(this->get_logger(), "No cameras found.");
    throw std::runtime_error("No cameras found.");
  }

  if (cameraId != nullptr)
  {
    err = m_vmbSystem.GetCameraByID(cameraId, m_camera);
    if (err != VmbErrorSuccess)
    {
      m_vmbSystem.Shutdown();
      RCLCPP_ERROR(this->get_logger(),
                   "No camera found with ID=%s, err = '%s'",
                   std::string(cameraId).data(),
                   std::to_string(err).data());
      throw std::runtime_error("No camera found with ID=" + std::string(cameraId) + ", err = " + std::to_string(err));
    }
  }
  else
  {
    m_camera = cameras[0];
  }

  err = m_camera->Open(VmbAccessModeFull);
  if (err != VmbErrorSuccess)
  {
    m_vmbSystem.Shutdown();
    RCLCPP_ERROR(this->get_logger(), "Could not open camera, err='%s'", std::to_string(err).data());
    throw std::runtime_error("Could not open camera, err=" + std::to_string(err));
  }

  std::string name;
  if (m_camera->GetName(name) == VmbErrorSuccess)
  {
    RCLCPP_INFO(this->get_logger(), "Opened Camera %s", name.data());
  }

  try
  {
    GigEAdjustPacketSize(m_camera);
  }
  catch (std::runtime_error &e)
  {
    m_vmbSystem.Shutdown();
    throw e;
  }

  // initialize the frames
  try
  {
    VmbCPP::FeaturePtr feature;
    VmbInt64_t nPLS; // payloas size
    m_camera->GetFeatureByName("PayloadSize", feature);
    feature->GetValue(nPLS);
    // initialize the frame
    frame_top_.reset(new VmbCPP::Frame(nPLS));
    frame_bot_.reset(new VmbCPP::Frame(nPLS));

    // register the observer
    frame_top_->RegisterObserver(
        VmbCPP::IFrameObserverPtr(
            new FrameObserver(m_camera,
                              std::bind(&Cic5000Camera::frameCallback_top, this, std::placeholders::_1))));

    frame_bot_->RegisterObserver(
        VmbCPP::IFrameObserverPtr(
            new FrameObserver(m_camera,
                              std::bind(&Cic5000Camera::frameCallback_bot, this, std::placeholders::_1))));

    // announce frame
    m_camera->AnnounceFrame(frame_top_);
    m_camera->AnnounceFrame(frame_bot_);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "Could set frame size, err='%s'", std::to_string(err).data());
    throw std::runtime_error("Could set frame size, err=" + std::to_string(err));
  }

  // initialize feature observers to get events about when the camera is exposing
  ActivateNotification();
  RegisterEvents();

  VmbCPP::FeaturePtr pFeature;
  m_camera->StartCapture();
  RCLCPP_INFO(this->get_logger(), "Start capture");
  m_camera->GetFeatureByName("AcquisitionStart", pFeature);
  pFeature->RunCommand();

  // queue the frames we are goingto sue
  // m_camera->QueueFrame(frame_bot_);
  // m_camera->QueueFrame(frame_top_);
}

VmbErrorType Cic5000Camera::ActivateNotification()
{
  //    std::cout << "Activating notifications for 'AcquisitionStart' events.\n";
  RCLCPP_INFO(this->get_logger(), "Activating notifications for 'EventExposureEnd' events.");

  // EventSelector is used to specify the particular Event to control
  VmbCPP::FeaturePtr feature;
  VmbErrorType err = m_camera->GetFeatureByName("EventSelector", feature);

  if (err == VmbErrorSuccess)
  {
    // Configure the AcquisitionStart camera event.
    err = feature->SetValue("ExposureEnd");
    if (err == VmbErrorSuccess)
    {
      // EventNotification is used to enable/disable the notification of the event specified by EventSelector.
      err = m_camera->GetFeatureByName("EventNotification", feature);
      if (err == VmbErrorSuccess)
      {
        err = feature->SetValue("On");
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Error turning EventNotification On");
      }
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Error setting feature to 'EventExposureEnd'");
    }
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Error getting feature 'EventSelector'");
  }

  return err;
}

void Cic5000Camera::RegisterEvents()
{
  // std::cout << "Registering observer for 'EventAcquisitionStart' feature.\n";
  RCLCPP_INFO(this->get_logger(), "Registering observer for 'EventExposureEnd' feature.\n");

  // Each of the events listed in the EventSelector enumeration will have a corresponding event identifier feature.
  // This feature will be used as a unique identifier of the event to register the callback function.
  VmbErrorType err = m_camera->GetFeatureByName("EventExposureEnd", eventFeature);

  if (err == VmbErrorSuccess)
  {
    // register a callback function to be notified that the event happened
    err = eventFeature->RegisterObserver(VmbCPP::IFeatureObserverPtr(new EventObserver(std::bind(&Cic5000Camera::exposure_end_callback, this, std::placeholders::_1))));

    if (err != VmbErrorSuccess)
    {
      RCLCPP_ERROR(this->get_logger(), "Error registering observer for 'EventExposureEnd' feature.\n");
    }
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Error getting feature 'EventExposureEnd'");
  }
}

void Cic5000Camera::Start()
{
  VmbErrorType err = m_camera->StartContinuousImageAcquisition(5, VmbCPP::IFrameObserverPtr(new FrameObserver(m_camera, std::bind(&Cic5000Camera::stream_callback, this, std::placeholders::_1))));
  if (err != VmbErrorSuccess)
  {
    RCLCPP_ERROR(this->get_logger(), "Could not start acquisition, err='%s'", std::to_string(err).data());
    throw std::runtime_error("Could not start acquisition, err=" + std::to_string(err));
  }
}

void Cic5000Camera::Stop()
{
  VmbErrorType err = m_camera->StopContinuousImageAcquisition();
  if (err != VmbErrorSuccess)
  {
    RCLCPP_ERROR(this->get_logger(), "Could not stop acquisition, err='%s'", std::to_string(err).data());
    throw std::runtime_error("Could not stop acquisition, err=" + std::to_string(err));
  }
}

void Cic5000Camera::CaptureSequence()
{


  // start reseting the plc
  auto reset_request = std::make_shared<cherry_interfaces::srv::ResetLatches::Request>();

  // setup the first fromae of the camera
  VmbCPP::FeaturePtr pFeature;
  // queue the frames we are goingto sue
  m_camera->QueueFrame(frame_bot_);

  reset_request->frame_id = frame_id_;
  reset_request->top_light = 0;
  reset_request->bot_light = 1;
  auto reset_result = plc_client_reset_latches_->async_send_request(reset_request);

  if (VmbErrorSuccess == m_camera->GetFeatureByName("Gain", pFeature))
  {
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Could not get gain feature!");
  }

  if (VmbErrorSuccess == pFeature->SetValue(bot_gain_))
  {
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Could not set gain feature!");
  }

  if (VmbErrorSuccess == m_camera->GetFeatureByName("TriggerSoftware", pFeature))
  {
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Could not get trigger feature!");
  }

  if (reset_result.wait_for(1000ms) == std::future_status::ready)
  {
    RCLCPP_INFO(this->get_logger(), "PLC is reset");
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Timeout waiting for PLC reset");
  }

  std::mutex mtx_triggerReady2;
  std::unique_lock<std::mutex> lck_triggerReady2(mtx_triggerReady2);
  if (cv_trigger_ready_.wait_for(lck_triggerReady2, std::chrono::milliseconds(1000)) == std::cv_status::timeout)
  {
    RCLCPP_ERROR(this->get_logger(), "Trigger  ready timeout");
  }
  else
  {
    RCLCPP_INFO(this->get_logger(), "Trigger ready for bot");
  }

    // note the position of the live encoder value
  // start_bot_ = count_;
  // start_bot_mm_ = count_mm_;

  // trigger!!
  VmbErrorType err = pFeature->RunCommand();
  if (err == VmbErrorSuccess)
  {
    RCLCPP_INFO(this->get_logger(), "Triggered bottom frame!");
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Error triggering bottom frame!");
  }



  std::mutex mtx_bot;
  std::unique_lock<std::mutex> lck_bot(mtx_bot);

  if (cv_frame_bot_.wait_for(lck_bot, std::chrono::milliseconds(1000)) == std::cv_status::timeout)
  {
    RCLCPP_ERROR(this->get_logger(), "Frame callback timeout");
  }
  else
  {
    RCLCPP_INFO(this->get_logger(), "Top notified");
  }

  m_camera->QueueFrame(frame_top_);

  // start reseting the plc
  auto light_request = std::make_shared<cherry_interfaces::srv::SetLights::Request>();

  light_request->top_light = 1;
  light_request->bot_light = 0;
  auto light_result = plc_client_set_lights_->async_send_request(light_request);

  // std::mutex mtx_triggerNotReady;
  // std::unique_lock<std::mutex> lck_triggerNotReady(mtx_triggerNotReady);
  // if (cv_trigger_notready_.wait_for(lck_triggerNotReady, std::chrono::milliseconds(2000)) == std::cv_status::timeout)
  // {
  //   RCLCPP_ERROR(this->get_logger(), "Trigger  ready timeout");
  // }
  // else
  // {
  //   RCLCPP_INFO(this->get_logger(), "Trigger ready for top");
  // }

  if (VmbErrorSuccess == m_camera->GetFeatureByName("Gain", pFeature))
  {
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Could not get gain feature!");
  }

  // top frame
  if (VmbErrorSuccess == pFeature->SetValue(top_gain_))
  {
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Could not set gain feature!");
  }

  if (VmbErrorSuccess == m_camera->GetFeatureByName("TriggerSoftware", pFeature))
  {
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Could not get trigger feature!");
  }

  if (light_result.wait_for(1000ms) == std::future_status::ready)
  {
    RCLCPP_INFO(this->get_logger(), "Light is reset");
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Timeout waiting for lights");
  }

  std::mutex mtx_triggerReady;
  std::unique_lock<std::mutex> lck_triggerReady(mtx_triggerReady);
  if (cv_trigger_ready_.wait_for(lck_triggerReady, std::chrono::milliseconds(1000)) == std::cv_status::timeout)
  {
    RCLCPP_ERROR(this->get_logger(), "Trigger  ready timeout");
  }
  else
  {
    RCLCPP_INFO(this->get_logger(), "Trigger ready for top");
  }

  //     // note the position of the live encoder value
  // start_top_ = count_;
  // start_top_mm_ = count_mm_;

  // trigger!!
  err = pFeature->RunCommand();
  if (err == VmbErrorSuccess)
  {
    RCLCPP_INFO(this->get_logger(), "Triggered top frame!");
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Error triggering top frame!");
  }


  std::mutex mtx_top;
  std::unique_lock<std::mutex> lck_top(mtx_top);

  if (cv_frame_top_.wait_for(lck_top, std::chrono::milliseconds(1000)) == std::cv_status::timeout)
  {
    RCLCPP_ERROR(this->get_logger(), "Frame callback timeout");
  }
  else
  {
    RCLCPP_INFO(this->get_logger(), "Top notified");
  }
  //}
  // turn the lights off
  light_request->top_light = 0;
  light_request->bot_light = 0;
  auto light_result2 = plc_client_set_lights_->async_send_request(light_request);

  if (light_result2.wait_for(1000ms) == std::future_status::ready)
  {
    RCLCPP_INFO(this->get_logger(), "Lights are off");
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Timeout waiting for lights");
  }
}

void Cic5000Camera::frameCallback_top(const VmbCPP::FramePtr frame)
{
  RCLCPP_INFO(this->get_logger(), "Top frame received");

  sensor_msgs::msg::Image image_top = sensor_msgs::msg::Image();
  frameToImage(frame, image_top);
  publisher_image_top_->publish(image_top);
  result_->image_top = image_top;

  // if stored isn;t working... fall back to current count
  // result_->count_top = start_top_;
  // result_->mm_top = start_top_mm_;
  result_->count_top = stored_;
  result_->mm_top = stored_mm_;

  cv_frame_top_.notify_one();
}
void Cic5000Camera::frameCallback_bot(const VmbCPP::FramePtr frame)
{
  RCLCPP_INFO(this->get_logger(), "Bot frame received");

  sensor_msgs::msg::Image image_bot = sensor_msgs::msg::Image();
  frameToImage(frame, image_bot);
  publisher_image_bot_->publish(image_bot);
  result_->image_bot = image_bot;

  // result_->count_bot = start_bot_;
  // result_->mm_bot = start_bot_mm_;
  result_->count_bot = stored_;
  result_->mm_bot = stored_mm_;

  cv_frame_bot_.notify_one();
}
void Cic5000Camera::stream_callback(const VmbCPP::FramePtr frame)
{
  RCLCPP_INFO(this->get_logger(), "Streaming");

  m_camera->QueueFrame(frame);
}

void Cic5000Camera::exposure_end_callback(const VmbCPP::FeaturePtr &feature)
{
  if (feature != nullptr)
  {
    std::string featureName("");
    VmbInt64_t featureValue;

    feature->GetName(featureName);
    feature->GetValue(featureValue);

    std::cout << "\nEvent " << featureName << " occurred. Value:" << featureValue << "\n";
    RCLCPP_INFO(this->get_logger(), "Event %s occurred, value = %lld", featureName.data(), featureValue);
  }

  RCLCPP_INFO(this->get_logger(), "exposure_end_callback called");
}

// int main(int argc, char **argv)
// {
//   rclcpp::init(argc, argv);

//   auto action_server = std::make_shared<Cic5000Camera>();

//   rclcpp::spin(action_server);

//   rclcpp::shutdown();
//   return 0;
// }
