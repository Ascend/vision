#include <Python.h>

#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/Exceptions.h>
#include <torch/library.h>
#include <ATen/ATen.h>

PyObject* module;
extern "C"

void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods)
{
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

static std::vector<PyMethodDef> methods;

static PyMethodDef TorchNpuOps[] = {
  {"tensor_add", (PyCFunction)TensorAdd, METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* getTorchNpuOps(){
    return TorchNpuOps;
}

PyObject* initMoudle(){
  AddPyMethodDefs(methods, getTorchNpuOps());
  static struct PyModuleDef torchnpu_module = {
     PyModuleDef_HEAD_INIT,
     "torchvision_npu.ops",
     nullptr,
     -1,
     methods.data()
  };
  module = PyModule_Create(&torchnpu_module);
  return module;
}

PyMODINIT_FUNC PyInit_ops(void){
    return initMoudle();
}