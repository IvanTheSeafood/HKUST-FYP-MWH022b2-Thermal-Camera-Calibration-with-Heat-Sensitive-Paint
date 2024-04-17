# Heat Sensitive Paint forThermal Camera Calibration


Final Year Project in 2023 to develop a calibration setup for thermal camreas using heat sensitive paint.

## Description

To create a thermal camera thermometer system affordable by the average consumer to improve public health safety, 
we developed a low-cost calibration system for thermal cameras such that they could accurately read body temperatures 
of different people in any environment by utilizing heat sensitive paint, which changes colour at a certain temperature.  
We applied OpenCV libraries using Python programming language and used Raspberry Pi to run our low-end thermal camera.  
We also utilized image processing to read our calibrator, resulting in a calibration system that can accurately measure 
each person’s body temperature in sight with minimum cost and set up complexity. Users only have to stand next to the 
calibrator in view of the thermal camera to have their body temperature read automatically by the camera system. 

## Getting Started

### Dependencies

* Webcam
* XLR90641 with 4 I/O pins
* Raspberry Pi 3B or above
* Windows 10 or above

### Installing

* Download the Pi.py code from the Raspberry Pi
* Modify Line 8 of Pi.py into your PC's IP address
* Download the Computer.py code from the Windows PC

### Executing program

* Connect Raspberry Pi to the Windows PC via wifi hotspot
* Run Pi.py on the Raspberry Pi
* Run Computer.py on Windows PC
* Position Calibrator within the yellow bounded square in the RGB image output
* Position people in view of the camera for measurements to take place

## Help

* If tracker loses the calibrator, press 'e' on the Windows PC keyboard to manually update the tracker program

## Authors

* LI Hoi Him (20585898)
* LI Hong Yin (20496853)
* LO Yat Hei (20692574)

## Acknowledgments

* [MakerPortal](https://makersportal.com/blog/2020/6/8/high-resolution-thermal-camera-with-raspberry-pi-and-mlx90640)
* [TowardsDataScience](https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81)
