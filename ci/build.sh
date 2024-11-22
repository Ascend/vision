#!/bin/bash

set -e

# 默认使用 python3
py_execute=python3

# 解析命令行参数
for arg in "$@"
do
    case $arg in
        --python=*)
            py_version="${arg#*=}"
            py_execute="python${py_version}"
            ;;
        *)
            # 对于非 --python 的参数打印警告
            echo "WARNING: Ignoring unsupported parameter: $arg"
            ;;
    esac
done

CUR_DIR=$(dirname $(readlink -f $0))

function main()
{
    cd ${CUR_DIR}/..

    $py_execute setup.py build bdist_wheel
    if [ $? != 0 ]; then
        echo "Failed to compile the wheel file. Please check the source code by yourself."
        exit 1
    fi

    exit 0
}

main "$@"
