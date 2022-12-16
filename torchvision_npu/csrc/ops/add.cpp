#include <Python.h>

#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/Exceptions.h>
#include <torch/library.h>

#include <ATen/ATen.h>

#include "torchvision_npu/csrc/ops/add.h"

static PyObject * TensorAdd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "tensor_add(Tensor a, Tensor b)",
  }, /*traceable=*/true);
  torch::ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);

  return torch::autograd::utils::wrap(_r.tensor(0).add(_r.tensor(1)));

  END_HANDLE_TH_ERRORS
}

