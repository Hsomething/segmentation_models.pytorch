from .model import SegmentationModel
from .ocr import OCR
from .modules import (
    Conv2dReLU,
    Attention,
)

from .heads import (
    SegmentationHead,
    ClassificationHead,
)
headers = {
    "common":SegmentationHead,
    "ocr":OCR
}
def get_header(header_name = "common",in_channels = 256,out_channels=512,activation = None,kernel_size = 3):
    try:
        header = headers[header_name](
            in_channels,out_channels,kernel_size,activation
        )
        return header
    except KeyError:
        raise KeyError("Wrong header name `{}`, supported headers: {}".format(header_name, list(headers.keys())))