# Copyright (c) 2024, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import subprocess
import threading
import queue
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch_npu


BASE_DIR = Path(__file__).absolute().parent.parent
TEST_DIR = os.path.join(BASE_DIR, 'test')

EXEC_TIMEOUT = 2000


class AccurateTest(metaclass=ABCMeta):
    base_dir = BASE_DIR

    @abstractmethod
    def identify(self, modify_file):
        """
        This interface provides the path information for the corresponding unit tests in the code.
        """
        raise Exception("abstract method. Subclasses should implement it.")

    @staticmethod
    def find_ut_by_regex(regex, test_path):
        ut_files = []
        cmd = "find {} -name {}".format(test_path, regex)
        status, output = subprocess.getstatusoutput(cmd)
        # For the ones that cannot be found, no action will be taken temporarily.
        if status or not output:
            pass
        else:
            files = output.split('\n')
            for ut_file in files:
                if ut_file.endswith(".py"):
                    ut_files.append(ut_file)
        return ut_files


class TestFileStrategy(AccurateTest):
    """
    Determine whether the modified files are test cases
    """
    def identify(self, modify_file):
        is_test_file = str(Path(modify_file).parts[0]) == "test" \
                       and re.match("test_(.+).py", Path(modify_file).name)
        return [(str(self.base_dir.joinpath(modify_file)))] if is_test_file else []


class CoreTestStrategy(AccurateTest):
    """
    Determine whether the core tests should be runned
    """
    def __init__(self):
        super().__init__()
        self.block_list = ['test', 'docs']
        self.core_test_cases = [str(i) for i in (self.base_dir / 'test/test_npu').rglob('test_*.py')]

    def identify(self, modify_file):
        modified_module = str(Path(modify_file).parts[0])
        if modified_module not in self.block_list:
            return self.core_test_cases
        return []


class TestMgr():
    def __init__(self):
        self.modify_files = []
        self.test_files = {
            'ut_files': [],
        }

    def load(self, modify_files):
        check_dir_path_readable(modify_files)
        with open(modify_files) as f:
            for line in f:
                line = line.strip()
                self.modify_files.append(line)

    def analyze(self):
        for modify_file in self.modify_files:
            self.test_files['ut_files'] += TestFileStrategy().identify(modify_file)
            self.test_files['ut_files'] += CoreTestStrategy().identify(modify_file)
        unique_files = sorted(set(self.test_files['ut_files']))

        exist_ut_file = [
            changed_file
            for changed_file in unique_files
            if Path(changed_file).exists()
        ]
        self.test_files['ut_files'] = exist_ut_file
        print(self.test_files['ut_files'])

    def get_test_files(self):
        return self.test_files

    def print_modify_files(self):
        print("modify files:")
        for modify_file in self.modify_files:
            print(modify_file)

    def print_ut_files(self):
        print("ut files:")
        for ut_file in self.test_files['ut_files']:
            print(ut_file)


def exec_ut(files):
    """
    Execute the unit test file, and if there are any failures, identify
        the exceptions and print relevant information.
    """
    def get_op_name(ut_file):
        return ut_file.split('/')[-1].split('.')[0].lstrip('test_')
    
    def get_ut_name(ut_file):
        return str(Path(ut_file).relative_to(TEST_DIR))[:-3]

    def get_ut_cmd(ut_type, ut_file):
        cmd = [sys.executable, "run_test.py", "-v", "-i"]
        return cmd + [get_ut_name(ut_file)]

    def wait_thread(process, event_timer):
        process.wait()
        event_timer.set()

    def enqueue_output(out, log_queue):
        for line in iter(out.readline, b''):
            log_queue.put(line.decode('utf-8'))
        out.close()
        return

    def start_thread(fn, *args):
        stdout_t = threading.Thread(target=fn, args=args)
        stdout_t.daemon = True
        stdout_t.start()

    def print_subprocess_log(log_queue):
        while (not log_queue.empty()):
            print((log_queue.get()).strip())

    def run_cmd_with_timeout(cmd):
        os.chdir(str(TEST_DIR))
        stdout_queue = queue.Queue()
        event_timer = threading.Event()

        p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        start_thread(wait_thread, p, event_timer)
        start_thread(enqueue_output, p.stdout, stdout_queue)

        try:
            event_timer.wait(EXEC_TIMEOUT)
            ret = p.poll()
            if ret:
                print_subprocess_log(stdout_queue)
            if not event_timer.is_set():
                ret = 1
                parent_process = psutil.Process(p.pid)
                for children_process in parent_process.children(recursive=True):
                    children_process.kill()
                p.kill()
                p.terminate()
                print("Timeout: Command '{}' timed out after {} seconds".format(" ".join(cmd), EXEC_TIMEOUT))
                print_subprocess_log(stdout_queue)
        except Exception as err:
            ret = 1
            print(err)
        return ret

    def run_tests(files):
        exec_infos = []
        has_failed = 0
        for ut_type, ut_files in files.items():
            for ut_file in ut_files:
                cmd = get_ut_cmd(ut_type, ut_file)
                ut_info = " ".join(cmd[4:]).replace(" -- -k", "")
                print("start running ut {}: ".format(ut_info))
                ret = run_cmd_with_timeout(cmd)
                if ret:
                    has_failed = ret
                    exec_infos.append("exec ut {} failed.".format(ut_info))
                else:
                    exec_infos.append("exec ut {} success.".format(ut_info))
        return has_failed, exec_infos

    ret_status, exec_infos = run_tests(files)

    print("***** Total result:")
    for exec_info in exec_infos:
        print(exec_info)
    return ret_status


def check_dir_path_readable(file_path):
    """
    check file path readable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path does not exist: {file_path}")
    if os.stat(file_path).st_uid != os.getuid():
        check_msg = input("The path does not belong to you, do you want to continue? [y/n]")
        if check_msg.lower() != 'y':
            raise RuntimeError("The user choose not to contiue")
    if os.path.islink(file_path):
        raise RuntimeError(f"Invalid path is a soft chain: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"The path permission check filed: {file_path}")


if __name__ == "__main__":
    cur_modify_files = os.path.join(BASE_DIR, 'modify_files.txt')
    test_mgr = TestMgr()
    test_mgr.load(cur_modify_files)
    test_mgr.analyze()
    cur_test_files = test_mgr.get_test_files()

    test_mgr.print_modify_files()
    test_mgr.print_ut_files()

    ret_ut = exec_ut(cur_test_files)
    sys.exit(ret_ut)
