# Apply-Kalman-Filter-and-Particles-Filter-to-detect-the-ball-movement

During this project, I applied two methods, Kalman Filter and Particle Filter, on two videos to detect the blue ball. The goal is to compare the difference between two methods. 

On Kalman Filter method, I need to set up a cv2.KalmanFilter function to measure and transit the matrix. And then I need to set a python Class to process the input video, like applying cv2.VideoCapture function to get the videos, after that I utilized cv2.circle, cv2.line and cv2.putText to draw the actual cords and expected cords predicted by Kalman Filter. By comparing two different cords, the model could represent how well the Kalman Filter performs.   The essential step is to detect the blue ball from input videos.  During the DetectBall function, I first need to convert the color format from BGR to HSV, however in order to catch blue color, the lower bound and the upper bound of blue color of HSV need to be defined. By searching a color diagram, blue was set from [100,100,20] to [130,255,255]. Eventually, cv2.circle helped me to outline the blue ball. 

 

The two videos are all about blue ball, but existing some difference, for example, one video is about a blue ball drop down from a stair and then rebound with cartoon design and distinguishable background color. Another video is about a dog playing with a blue ball, and it is real world colors, so sometimes it is a little bit difficult to distinguish blue ball from shadows. 

Kalman Filter performed better in the first video, because the blue color is easy to distinguish from background color, on the other hand, the ball is totally a circle.  However, Due to the unnatural rebound, Kalman Filter couldn’t predict the ball movement perfectly with a little difference. 
 


