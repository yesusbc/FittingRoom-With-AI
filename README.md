# FittingRoom-with-AI

Tool to try on different t-shirts, using pose estimation with the OpenVINO toolkit 

## Getting Started

For running the app you must have the OpenVINO toolkit downloaded.

Inside your workspace directory you must have a tree like this one:

├── app.py
├── helper.py
├── image (Input images, yours and the different t-shirts you want to try on)
│   ├── human.jpeg
│   ├── tshirt1.jpg
│   └── tshirt2.jpg
├── inference.py
├── intel
│   └── human-pose-estimation-0001 (The human pose models)
│       ├── FP16
│       │   ├── human-pose-estimation-0001.bin
│       │   └── human-pose-estimation-0001.xml
│       ├── FP32
│       │   ├── human-pose-estimation-0001.bin
│       │   └── human-pose-estimation-0001.xml
│       └── FP32-INT8
│           ├── human-pose-estimation-0001.bin
│           └── human-pose-estimation-0001.xml
├── outputs
│   ├── (Images with your t-shirt putted on)
│   └── tshirt_try_on.png
└── t_shirt_coords.py

About the files:
* App.py contains the main operations (Human pose estimation, and combination with the tshirt)
* t_shirt_coords.py is for getting the important coordinates of the t-shirt
* inference.py is for working with the inference engine
* helper.py are helper functions for working with the pse estimation

### Prerequisites

OpenVINO toolkit
Image of the Model
Image of the t-shirt/polo/etc

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

To test your implementations, you can use app.py to run each edge application, with the following arguments:

-t: The model type, which should be one of "POSE", "TEXT", or "CAR_META"
-m: The location of the model .xml file
-i: The location of the input image used for testing
-c: A CPU extension file, if applicable. See below for what this is for the workspace. The results of your output will be saved down for viewing in the outputs directory.
As an example, here is an example of running the app with related arguments

python app.py -i "images/blue-car.jpg" -t "CAR_META" -m "/home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"



### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
