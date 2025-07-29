// Copyright (c) 2025, Huawei Technologies.All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <torch_npu/csrc/core/npu/GetCANNInfo.h>
#include "pytorch_npu_helper.hpp"

constexpr size_t kVersionIndex1 = 1;
constexpr size_t kVersionIndex2 = 2;
constexpr size_t kVersionIndex3 = 3;
constexpr size_t kVersionIndex4 = 4;

int64_t CannVersionToNum(const std::string &versionStr)
{
    std::smatch results;
    int64_t major = -1;
    int64_t minor = -1;
    int64_t release = -1;
    int64_t RCVersion = -51;
    int64_t TVersion = -1;
    int64_t alphaVersion = 0;
    if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+)"))) {
        major = stoll(results[kVersionIndex1]);
        minor = stoll(results[kVersionIndex2]);
        RCVersion = stoll(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).([0-9]+)"))) {
        major = stoll(results[kVersionIndex1]);
        minor = stoll(results[kVersionIndex2]);
        release = stoll(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).T([0-9]+)"))) {
        major = stoll(results[kVersionIndex1]);
        minor = stoll(results[kVersionIndex2]);
        TVersion = stoll(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+).alpha([0-9]+)"))) {
        major = stoll(results[kVersionIndex1]);
        minor = stoll(results[kVersionIndex2]);
        RCVersion = stoll(results[kVersionIndex3]);
        alphaVersion = stoll(results[kVersionIndex4]);
    }

    int64_t num = ((major + 1) * 100000000) +
                 ((minor + 1) * 1000000) +
                 ((release + 1) * 10000) +
                 ((RCVersion + 1) * 100 + 5000) +
                 ((TVersion + 1) * 100) - (100 - alphaVersion);
    return num;
}

bool IsGteCANNVersion(const std::string &version)
{
    std::string currentVersion = GetCANNVersion();
    int64_t current_num = CannVersionToNum(currentVersion);
    int64_t boundary_num = CannVersionToNum(version);
    if (current_num >= boundary_num) {
        return true;
    } else {
        return false;
    }
}
