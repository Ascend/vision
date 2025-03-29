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

double CannVersionToNum(const std::string &versionStr)
{
    std::smatch results;
    int major = -1;
    int minor = -1;
    int release = -1;
    int RCVersion = -51;
    int TVersion = -1;
    int alphaVersion = 0;
    if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+)"))) {
        major = stoi(results[1]);
        minor = stoi(results[2]);
        RCVersion = stoi(results[3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).([0-9]+)"))) {
        major = stoi(results[1]);
        minor = stoi(results[2]);
        release = stoi(results[3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).T([0-9]+)"))) {
        major = stoi(results[1]);
        minor = stoi(results[2]);
        TVersion = stoi(results[3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+).alpha([0-9]+)"))) {
        major = stoi(results[1]);
        minor = stoi(results[2]);
        RCVersion = stoi(results[3]);
        alphaVersion = stoi(results[4]);
    }

    double num = ((major + 1) * 100000000) + ((minor + 1) * 1000000) + ((release + 1) * 10000) + ((RCVersion + 1) * 100 + 5000) + ((TVersion + 1) * 100) - (100 - alphaVersion);
    return num;
}

bool IsGteCANNVersion(const std::string &version)
{
    std::string currentVersion = GetCANNVersion();
    double current_num = CannVersionToNum(currentVersion);
    double boundary_num = CannVersionToNum(version);
    if (current_num >= boundary_num) {
        return true;
    } else {
        return false;
    }
}
