#

# Copyright (c) 2014-2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

CTS_VERSION_FILE := $(strip $(wildcard $(LOCAL_PATH)/../../openvx_cts_version.inc))
ifdef CTS_VERSION_FILE
    LOCAL_CFLAGS += -DHAVE_VERSION_INC
endif

FILE_LIST := $(wildcard $(LOCAL_PATH)/../../test_engine/*.c) $(wildcard $(LOCAL_PATH)/../../test_conformance/*.c)
LOCAL_SRC_FILES := $(FILE_LIST:$(LOCAL_PATH)/%=%)
LOCAL_C_INCLUDES := $(OPENVX_INCLUDES) $(LOCAL_PATH)/../../ $(LOCAL_PATH)/../../test_conformance
LOCAL_LDLIBS := $(OPENVX_LIBRARIES)
LOCAL_MODULE := vx_test_conformance
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
    LOCAL_CFLAGS += -DHAVE_NEON=1 -march=armv7-a -mfpu=neon -ftree-vectorize -ffast-math -mfloat-abi=softfp
endif
ifneq ($(CT_DISABLE_TIME_SUPPORT),1)
    LOCAL_CFLAGS += -DCT_TEST_TIME
endif
$(info ${LOCAL_CFLAGS})
include $(BUILD_EXECUTABLE)
