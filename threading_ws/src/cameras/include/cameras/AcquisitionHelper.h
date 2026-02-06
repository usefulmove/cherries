/*=============================================================================
  Copyright (C) 2012-2023 Allied Vision Technologies.  All Rights Reserved.
  Subject to the BSD 3-Clause License.
=============================================================================*/

#ifndef ASYNCHRONOUS_GRAB_H_
#define ASYNCHRONOUS_GRAB_H_

#include <VmbCPP/VmbCPP.h>
  #include "rclcpp/rclcpp.hpp"

class AcquisitionHelper
{
private:
    VmbCPP::VmbSystem&  m_vmbSystem;
    VmbCPP::CameraPtr   m_camera;
    // rclcpp::Node   node_;
    // std::function<void(std::string)> logger;

public:
    /**
     * \brief The constructor will initialize the API and open the given camera
     *
     * \param[in] pCameraId  zero terminated C string with the camera id for the camera to be used
     */
    AcquisitionHelper(const char* cameraId);

    /**
     * \brief The constructor will initialize the API and open the first available camera
     */
    AcquisitionHelper();

  //   /**
  //  * \brief give the Helper acces to a node so that it can log stufff
  //  */
  //   AcquisitionHelper(std::function<void(std::string)> func);

    /**
     * \brief The destructor will stop the acquisition and shutdown the API
     */
    ~AcquisitionHelper();

    /**
     * \brief Start the acquisition.
     */
    void Start();

    /**
     * \brief Stop the acquisition.
     */
    void Stop();
};


#endif
