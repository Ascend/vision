#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

std::pair<double, double> Transform(double x, double y, std::vector<double>& matrix)
{
    return std::make_pair(matrix[0] * x + matrix[1] * y + matrix[2],
                          matrix[3] * x + matrix[4] * y + matrix[5]);
}

c10::SmallVector<int64_t, SIZE> rotate_output_size(
    const at::Tensor& self,
    bool expand,
    double angle)
{
    int64_t input_h = self.size(2);
    int64_t input_w = self.size(3);

    int64_t output_h = input_h;
    int64_t output_w = input_w;

    if (expand) {
        const double pi = std::acos(-1);
        angle = std::fmod(angle, 360.0);
        const double radians = -(angle / 180 * pi);

        const double precision = std::pow(10, 15);
        TORCH_CHECK(precision != 0, "Zero division error, precision=0.");
        std::vector<double> matrix = {
            std::round(std::cos(radians) * precision) / precision,
            std::round(std::sin(radians) * precision) / precision,
            0.0,
            std::round(-std::sin(radians) * precision) / precision,
            std::round(std::cos(radians) * precision) / precision,
            0.0,
        };

        std::pair<double, double> rotn_center = {input_w / 2.0, input_h / 2.0};

        auto tmp = Transform(-rotn_center.first, -rotn_center.second, matrix);
        matrix[2] = tmp.first;
        matrix[5] = tmp.second;

        matrix[2] += rotn_center.first;
        matrix[5] += rotn_center.second;

        std::vector<std::pair<double, double>> points = {
            {0, 0},
            {input_w, 0},
            {input_w, input_h},
            {0, input_h},
        };

        auto f = [&matrix](std::pair<double, double>& p) {
        p = Transform(p.first, p.second, matrix);
        return p;
        };
        std::transform(points.begin(), points.end(), points.begin(), f);

        auto x_comp = [](auto& p0, auto& p1) {return p0.first < p1.first;};
        auto x_max = std::max_element(points.cbegin(), points.cend(), x_comp);
        auto x_min = std::min_element(points.cbegin(), points.cend(), x_comp);

        auto y_comp = [](auto& p0, auto& p1) {return p0.second < p1.second;};
        auto y_max = std::max_element(points.cbegin(), points.cend(), y_comp);
        auto y_min = std::min_element(points.cbegin(), points.cend(), y_comp);

        output_h = std::ceil(y_max->second) - std::floor(y_min->second);
        output_w = std::ceil(x_max->first) - std::floor(x_min->first);
    }

    c10::SmallVector<int64_t, SIZE> output_size = {self.size(0), self.size(1), output_h, output_w};
    
    return output_size;
}

at::Tensor rotate_aclnn_kernel(
    const at::Tensor& self,
    double angle,
    int64_t interpolation_mode,
    bool expand,
    at::IntArrayRef center,
    int64_t padding_mode,
    c10::optional<c10::ArrayRef<double>> fill)
{
    float angle_cast = static_cast<float>(angle);

    std::vector<float> f_vec;
    if (fill.has_value()) {
        TORCH_CHECK(fill.value().size() == 3, "Param[fill] size should be 3.");
        f_vec = array_to_vector_cast<float, double>(fill.value());
    } else {
        f_vec = {0, 0, 0};
    }
    at::ArrayRef<float> fill_cast(f_vec);

    auto output_size = rotate_output_size(self, expand, angle);
    at::Tensor result = at::empty(output_size, self.options());
    EXEC_NPU_CMD(acldvppRotate, self, angle_cast, interpolation_mode, expand, center, padding_mode, fill_cast, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_rotate_aclnn"), TORCH_FN(rotate_aclnn_kernel));
}

} // namespace ops
} // namespace vision
