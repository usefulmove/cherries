/*=============================================================================
  Copyright (C) 2012-2023 Allied Vision Technologies.  All Rights Reserved.
  Subject to the BSD 3-Clause License.
=============================================================================*/

#include <iostream>
#include "EventObserver.h"

EventObserver::EventObserver(FeatureCallback callback) : VmbCPP::IFeatureObserver(), callback_(callback)
{
}

void EventObserver::FeatureChanged(const VmbCPP::FeaturePtr &feature)
{
  callback_(feature);
}