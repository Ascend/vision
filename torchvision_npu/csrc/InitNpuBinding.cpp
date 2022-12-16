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

PyObject * FirstFunction(PyObject * /* unused */)
{
  fprintf(stdout, "call c function success.\n");
  Py_RETURN_NONE;
}

//PyObject * TensorAdd(PyObject* self_, PyObject* args, PyObject* kwargs)
//{
//  fprintf(stdout, "call c function success.\n");
//  Py_RETURN_NONE;
//}

static PyObject * TensorAdd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "tensor_add(Tensor a, Tensor b)",
  }, /*traceable=*/true);
  torch::ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);

  return torch::autograd::utils::wrap(_r.tensor(0).add(_r.tensor(1)));

//  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}


static PyMethodDef TorchNpuMethods[] = {
  {"FirstFunction", (PyCFunction)FirstFunction, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

static PyMethodDef TorchNpuOps[] = {
  {"tensor_add", (PyCFunction)TensorAdd, METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

static std::vector<PyMethodDef> methods;

PyObject* initMoudle(){
  AddPyMethodDefs(methods, TorchNpuMethods);
  AddPyMethodDefs(methods, TorchNpuOps);
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