/**
 * \brief IFrameObserver implementation for asynchronous image acquisition
 */

#include "frameObserver.hpp"

FrameObserver::FrameObserver(VmbCPP::CameraPtr camera, Callback callback) : 
VmbCPP::IFrameObserver(camera), 
cam_ptr_(camera), 
callback_(callback)
{
    //callback_ = callback;
};

void FrameObserver::FrameReceived(const VmbCPP::FramePtr frame)
{
    //   VmbFrameStatusType eReceiveStatus;
    //   VmbErrorType err = frame->GetReceiveStatus(eReceiveStatus);

    //   if (err == VmbErrorSuccess)
    //   {
    //     switch (eReceiveStatus)
    //     {
    //       case VmbFrameStatusComplete: {
    //         // Call the callback
    //         //std::cout << "recieved frame!" << std::endl;
    //         //RCLCPP_INFO(this->get_logger(), "recieved frame!");
    //         //callback_(vimba_frame_ptr);
    //         break;
    //       }
    //       case VmbFrameStatusIncomplete: {
    //         //std::cout << "ERR: FrameObserver VmbFrameStatusIncomplete" << std::endl;
    //         //RCLCPP_ERROR(this->get_logger(), "FrameObserver VmbFrameStatusIncomplete.");
    //         break;
    //       }
    //       case VmbFrameStatusTooSmall: {
    //         //std::cout << "ERR: FrameObserver VmbFrameStatusTooSmall" << std::endl;
    //         //CLCPP_ERROR(this->get_logger(), "FrameObserver VmbFrameStatusTooSmall.");
    //         break;
    //       }
    //       case VmbFrameStatusInvalid: {
    //         //std::cout << "ERR: FrameObserver VmbFrameStatusInvalid" << std::endl;
    //         //RCLCPP_ERROR(this->get_logger(), "FrameObserver VmbFrameStatusInvalid.");
    //         break;
    //       }
    //       default: {
    //         //std::cout << "ERR: FrameObserver no known status" << std::endl;
    //         //RCLCPP_ERROR(this->get_logger(), "FrameObserver no known status.");
    //         break;
    //       }
    //     }
    //   }

    // always call back
    callback_(frame);
};