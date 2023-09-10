#ifndef LYJ_SRC_UTIL_LOG_H_
#define LYJ_SRC_UTIL_LOG_H_

#ifndef BUILD_ANDROID
#include <glog/logging.h>
#else
#include "logging.h"
#endif

#include <iostream>

#define AINFO LOG(INFO)
#define AWARN LOG(WARNING)
#define AERROR LOG(ERROR)
#define AFATAL LOG(FATAL)

#endif  // LYJ_SRC_UTIL_LOG_H_