This repo is NOT actively maintained and may not work out of the box as it has been 3 years since the last update. If you want to learn more about the next version of this project, check it out here: https://www.youtube.com/watch?v=tZcRYcjTwWA.


<h1>Phormatics: <em>Using AI to Maximize Your Workout</em></h1>

<img src="https://github.com/jrobchin/phormatics/blob/master/screenshots/frontpage.gif?raw=true" height="250px"></br>
<sup><em>f1: front page (the gif may be choppy at first, but it's worth it I promise)</em></sup>

by: Jason Chin <a href="https://linkedin.com/in/jrobchin"><img src="https://raw.githubusercontent.com/jrobchin/phormatics/master/screenshots/linkedin.png" height="20px"></a> <a href="https://github.com/jrobchin"><img src="https://raw.githubusercontent.com/jrobchin/phormatics/master/screenshots/github.png" height="20px"></a>, Charlie Lin <a href="https://www.linkedin.com/in/charlielin10/"><img src="https://raw.githubusercontent.com/jrobchin/phormatics/master/screenshots/linkedin.png" height="20px"></a> <a href="https://l.facebook.com/l.php?u=https%3A%2F%2Fgithub.com%2Fcharlielin99&h=ATOwBl6A7WzoJrSLqEMOb8lND5QQHnHCDu2wzFA2GEPfggdX2nlD_IZgnMX_ybgADA5TsQl483yueldKeHhCoD_hxt6uDABDplBaSmxtMBlDLh291-WB6JjZ1UOiQQ"><img src="https://raw.githubusercontent.com/jrobchin/phormatics/master/screenshots/github.png" height="20px"></a>, Brad Huang <a href="https://linkedin.com/in/brad-huang"><img src="https://raw.githubusercontent.com/jrobchin/phormatics/master/screenshots/linkedin.png" height="20px"></a> <a href="https://github.com/BradHuang1999"><img src="https://raw.githubusercontent.com/jrobchin/phormatics/master/screenshots/github.png" height="20px"></a>, Calvin Woo <a href="https://www.linkedin.com/in/cwoozle/"><img src="https://raw.githubusercontent.com/jrobchin/phormatics/master/screenshots/linkedin.png" height="20px"></a> <a href="https://github.com/cwoozle"><img src="https://raw.githubusercontent.com/jrobchin/phormatics/master/screenshots/github.png" height="20px"></a>

[HackNYU2018](http://hacknyu.org/) project developed in 36 hours, focusing on using A.I. and computer vision to build a virtual personal fitness trainer. Capable of using 2D human pose estimation with commodity web-cameras to critique your form and count your repetitions.

This project won the award for "The Most Startup-Viable Hack" as awarded by [Contrary Capital](https://contrarycap.com/). 

### 2D Human Pose Estimation:
<img src="https://github.com/jrobchin/phormatics/blob/master/screenshots/usage.png?raw=true" height="335px"></img></br>
<sup><em>f2: live pose estimation in a busy environment; note: here the user has over-extended their right arm (image is mirrored), which is considered bad form in this variant of the dumb bell shulder press, hence the message.</em></sup>

The pose estimation was based off of [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) by [ildoonet](https://github.com/ildoonet). The model architecture, [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) developed by [CMU Perceptual Computing Lab](https://www.cmu.edu/), consists of a deep convolutional neural network for feature extraction ([MobileNet](https://arxiv.org/abs/1704.04861)) and a two-branch multi-stage CNN for confidence maps and Part Affinity Fields (PAFs). 

This feature allowed us to track the position of the user's joints using a commodity webcam.

### Data Flow (Web Based):
<img src="https://github.com/jrobchin/phormatics/blob/master/notes/dataflow_diagram.jpg?raw=true"></img>
<sup><em>f3: pseudo data flow diagram; note: the pose estimation model output must be processed as it returns pose estimation for all possible humans in frame (see: [Future Changes <sup>[1]</sup>](https://github.com/jrobchin/phormatics#future-changes)).</em></sup>

This app runs in browser and the pose estimation and form critique generation is performed on a [Flask](http://flask.pocoo.org/) server. The webcam feed is captured using [WebRTC](https://webrtc.org/) and screenshots are sent to the server as a base64 encoded string every 50ms or as fast as the server can respond - which ever is slower (see: [Future Changes <sup>[2]</sup>](https://github.com/jrobchin/phormatics#future-changes)). 

This means the server could be run in the cloud on high-performance hardware and the client could be any device with a WebRTC-supported web browser and camera. There is also the option for video to be recorded and sent to the server for post-processing if the user's network connectivity is too slow to stream a live feed.

### Currently Supported Exercise Analysis:

- Squat: *exaggerated knees-forward checking*
- Dumbbell Shoulder Press: *exaggerated arm bend and extension checking*
- Bicep Curls: *horizontal elbow deviation from shoulder checking*

### Future Changes:
1. **Multiple Pose Estimations for One User**	

	*Current:* 
	The model estimates joints for all subjects found in the input image; we then analyze the output and extract the pose that is most likely to be the user.

	*Possible Improvements:* 
	
	a. Modify model and training data to only estimate a single 'best' pose.
	
	or
	
	b. Implement re-identification and support multiple users at once. This is viable as forward propagation time does not increase with multiple poses being estimated.

2. **Webcam Image Data Transfer**	

	*Current:*
	Webcam captures are encoded in base64 strings and a post request is sent to the server with the data (*note: this was done for ease of implementation due to the hackathon time constraint*).

	*Possible Improvements:*
	Implement web sockets to transfer webcam captures instead.  
