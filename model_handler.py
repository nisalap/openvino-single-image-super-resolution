"""
Model Handler for Openvino models
"""
import os
import requests
import numpy as np
import cv2
from openvino.inference_engine import IECore


class ModelHandler:
    def __init__(self, model_xml):
        ie = IECore()
        # self.network = ie.read_network(model=model_xml, weights=model_bin)
        if not os.path.exists(model_xml):
            self.download_models()
        self.executable_network = ie.load_network(network=model_xml, device_name="CPU")
        self.inputs_required = self.executable_network.input_info
        for each in self.inputs_required.keys():
            batch, channels, height, width = self.executable_network.input_info[each].tensor_desc.dims
            self.inputs_required[each] = (batch, channels, height, width)

    def preprocess(self, frame):
        for each in self.inputs_required:
            n, c, h, w = self.inputs_required[each]
            # h, w, c
            frame = cv2.resize(src=frame, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            input_data = np.expand_dims(np.transpose(frame, (2, 0, 1)), 0).astype(np.float16)
            self.inputs_required[each] = input_data

    def inference(self, image):
        self.preprocess(image)
        output = self.executable_network.infer(self.inputs_required)
        # Output is an image
        output = output["90"]
        output = np.squeeze(output, axis=0)
        output = (np.transpose(output, (1, 2, 0)) * 255).astype(np.int)
        cv2.imwrite("output.png", output)
        return output

    @staticmethod
    def download_file(url, dir):
        response = requests.get(url=url, stream=True)
        file_name = url.split("/")[-1]
        if not os.path.exists(dir):
            os.mkdir(dir)
        if response.status_code == 200:
            new_file_name = os.path.join(dir, file_name)
            with open(new_file_name, 'wb') as f:
                for chunk in response:
                    f.write(chunk)
            return new_file_name

    def download_models(self):
        # URL: https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/single-image-super-resolution-1033/
        model_xml = "https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/single-image-super-resolution-1033/FP32/single-image-super-resolution-1033.xml"
        model_bin = "https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/single-image-super-resolution-1033/FP32/single-image-super-resolution-1033.bin"

        print(f"Downloaing: {model_bin}")
        self.download_file(url=model_bin, dir="bin")

        print(f"Downloaing: {model_xml}")
        file_name = self.download_file(url=model_xml, dir="bin")

        print(f"Downloaded models")


def test_ModelHandler():
    model = ModelHandler(model_xml="bin/single-image-super-resolution-1033.xml")
    frame = cv2.imread("test.png")
    model.inference(frame)


test_ModelHandler()
