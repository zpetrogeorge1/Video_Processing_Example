# Video Processing Example with Computer Vision
This repository is a demonstration of Video Processing using Google's MediaPipe

The project is a simple program that constantly pulls data from a web camera, and uses Google's MediaPipe handtracking model (hand_landmarker.task). 
The model can track 21 points on a hand, which can be extracted to identify the hand a live camera feed, as well as perform functions as the hands move to different areas of the screen.
This is because each landmark is constantly identified on an X,Y plan from 0 - 1. 

While this example simply shows the landmarks on the live feed for one hand, multiple hands can be extracted and the corresponding hand data points can be utilized.
