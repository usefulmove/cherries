/*=============================================================================
  Copyright (C) 2012-2023 Allied Vision Technologies.  All Rights Reserved.
  Subject to the BSD 3-Clause License.
=============================================================================*/

#ifndef EVENTOBSERVER_H
#define EVENTOBSERVER_H

#include <VmbCPP/VmbCPP.h>
#include <VmbCPP/IFeatureObserver.h>

class EventObserver : public VmbCPP::IFeatureObserver
{
public:
  typedef std::function<void(const VmbCPP::FeaturePtr &pFeature)> FeatureCallback;

  EventObserver(FeatureCallback callback);

  /**
   * \brief This function will be called when the observed camera feature has changed
   */
  virtual void FeatureChanged(const VmbCPP::FeaturePtr &pFeature);

private:
  FeatureCallback callback_;
};

#endif // EVENTOBSERVER_H