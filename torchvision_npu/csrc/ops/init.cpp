#include <Python.h>

#include "torchvision_npu/csrc/ops/add.h"

static PyMethodDef TorchNpuOps[] = {
  {"tensor_add", (PyCFunction)TensorAdd, METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* getTorchNpuOps(){
    return TorchNpuOps;
}
