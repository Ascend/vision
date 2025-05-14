#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {
namespace {
struct ImageShapeInfo {
    int64_t width;
    int64_t height;
    int64_t channels;
};

static const std::unordered_set<uint16_t> marker_sof = {
    0xFFC0, 0xFFC1, 0xFFC2, 0xFFC3, 0xFFC5, 0xFFC6, 0xFFC7,
    0xFFC9, 0xFFCA, 0xFFCB, 0xFFCD, 0xFFCE, 0xFFCF
};


ImageShapeInfo get_jpeg_shape_from_memory(const uint8_t* jpegData, size_t dataLength)
{
    ImageShapeInfo info = {0, 0, 0};
    size_t min_length = 2;
    if (dataLength < min_length || jpegData[0] != 0xFF || jpegData[1] != 0xD8) {
        TORCH_CHECK(false, "invalid JPEG");
        return info;
    }

    size_t index = 2;
    auto times = 0;
    size_t byte_size = 8;
    size_t precision_size = 5;
    while (index < dataLength - 1) {
        times += 1;
        if (jpegData[index] == 0xFF) {
            uint16_t marker = (jpegData[index] << byte_size) | jpegData[index + 1];
            if (marker_sof.find(marker) != marker_sof.end()) {
                // Skip precision byte
                index += precision_size;

                info.height = (jpegData[index] << byte_size) | jpegData[index + 1];
                index += 2;
                info.width = (jpegData[index] << byte_size) | jpegData[index + 1];
                index += 2;

                info.channels = jpegData[index];
                break;
            }
            if (index + 3 >= dataLength) {
                TORCH_CHECK(false, "no marker found");
                break;
            }
            int segmentLength = (jpegData[index + 2] << byte_size) | jpegData[index + 3];
            index += segmentLength + 2;
        }
        else {
            TORCH_CHECK(false, "invalid JPEG");
            break;
        }
    }
    return info;
}

std::tuple<int64_t, int64_t, int64_t> _get_jpeg_image_shape(const at::Tensor &data)
{
    TORCH_CHECK(data.dtype() == at::kByte, "Expected a torch.uint8 tensor");

    TORCH_CHECK(data.dim() == 1 && data.numel() > 0, "Expected a non empty 1-dimensional tensor");

    auto datap = data.data_ptr<uint8_t>();
    auto info = get_jpeg_shape_from_memory(datap, data.numel());
    return std::make_tuple(info.height, info.width, info.channels);
}

}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_get_jpeg_image_shape(Tensor self) -> (int ,int , int)"));
}

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_get_jpeg_image_shape"), TORCH_FN(_get_jpeg_image_shape));
}

}
}
