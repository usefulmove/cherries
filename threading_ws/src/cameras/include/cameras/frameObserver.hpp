
#ifndef FRAME_OBSERVER_H
#define FRAME_OBSERVER_H

#include <VmbCPP/VmbCPP.h>
#include <functional>

class FrameObserver : public VmbCPP::IFrameObserver
{
public:
    typedef std::function<void(const VmbCPP::FramePtr vimba_frame_ptr)> Callback;

    FrameObserver(VmbCPP::CameraPtr camera, Callback callback);
    virtual void FrameReceived(const VmbCPP::FramePtr frame);

private:
  // Frame observer stores all FramePtr
  VmbCPP::CameraPtr cam_ptr_;
  Callback callback_;

  double gain_;
  double exposure_;
  double index_;
};

#endif