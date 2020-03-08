# FittingRoom-with-AI

Tool to try on different t-shirts, using pose estimation with the OpenVINO toolkit 

This project was made for the Intel® Edge AI Scholarship Foundation Course Nanodegree Program.

## Getting Started

For running the app you must have the OpenVINO toolkit downloaded.

Inside your workspace directory you must have a tree like this one:

├── app.py <br />
├── helper.py <br />
├── image (Input images, yours and the different t-shirts you want to try on) <br />
│   ├── human.jpeg <br />
│   ├── tshirt1.jpg <br />
│   └── tshirt2.jpg <br />
├── inference.py <br />
├── intel <br />
│   └── human-pose-estimation-0001 (The human pose models) <br />
│       ├── FP16 <br />
│       │   ├── human-pose-estimation-0001.bin <br />
│       │   └── human-pose-estimation-0001.xml <br />
│       ├── FP32 <br />
│       │   ├── human-pose-estimation-0001.bin <br />
│       │   └── human-pose-estimation-0001.xml <br />
│       └── FP32-INT8 <br />
│           ├── human-pose-estimation-0001.bin <br />
│           └── human-pose-estimation-0001.xml <br />
├── outputs <br />
│   ├── (Images with your t-shirt putted on) <br />
│   └── tshirt_try_on.png <br />
└── t_shirt_coords.py <br />

About the files:
* App.py contains the main operations (Human pose estimation, and combination with the tshirt)
* t_shirt_coords.py is for getting the important coordinates of the t-shirt
* inference.py is for working with the inference engine
* helper.py are helper functions for working with the pse estimation

### Prerequisites

OpenVINO toolkit
Image of the Model
Image of the t-shirt/polo/etc

![Alt text](image/tshirt.jpg?raw=true "Image of the t-shirt")
![Alt text](image/human.jpeg?raw=true "Image of the Model", width="350" height="500")

### Installing

OpenVINO toolkit
https://software.intel.com/en-us/openvino-toolkit/choose-download?

## Running the tests

To test your implementations, you can use app.py to run each edge application, with the following arguments:

-m: The location of the model .xml file
-i: The location of the input image (model) used for testing
-s: The location of the input image (t-shirt) used for testing
-c (optional): A CPU extension file, if applicable. See below for what this is for the workspace. The results of your output will be saved down for viewing in the outputs directory.

As an example, here is an example of running the app with related arguments

python app.py -i "image/human.jpeg" -m "intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml" -s "image/tshirt.jpg"

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks for Mariana, for supporting me on the project
