# 

# Copyright (c) 2012-2017 The Khronos Group Inc.
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


include $(PRELUDE)
TARGET      := vx_test_conformance
TARGETTYPE  := exe
CSOURCES    := $(call all-c-files)
IDIRS       += $(HOST_ROOT)/cts
SHARED_LIBS := openvx vxu
STATIC_LIBS := vx_conformance_engine
ifneq (,$(findstring OPENVX_USE_NN_16,$(SYSDEFS)))
STATIC_LIBS +=  network
endif
ifeq ($(HOST_COMPILER),GCC)
CFLAGS += -Wno-unused-function
endif
include $(FINALE)
