

import cv2
import numpy as np

# set kalman filter
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, X, Y):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(X)], [np.float32(Y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted



# get ball coordinated in the video
class ProcessVideo:

    def DetectObject(self):

        vid = cv2.VideoCapture('/Users/ph/Desktop/ball6.mp4')

        if(vid.isOpened() is False):
            print('Cannot open input video')
            return

        width = int(vid.get(3))
        height = int(vid.get(4))

        # Create Kalman Filter Object
        kfObj = KalmanFilter()
        predictedCoords = np.zeros((2, 1), np.float32)

        while(vid.isOpened()):
            rc, frame = vid.read()

            if(rc is True):
                ballX, ballY,r = self.DetectBall(frame,5)
                predictedCoords = kfObj.Estimate(ballX, ballY)

                # Draw Actual coords 
                cv2.circle(frame, (ballX, ballY), 20, [0,0,255], 2, 8)
                cv2.line(frame,(ballX, ballY + 20), (ballX + 50, ballY + 20), [100,100,255], 2,8)
                cv2.putText(frame, "Actual", (ballX + 50, ballY + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])

                # Draw Kalman Filter Predicted output
                cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0,255,255], 2, 8)
                cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
                cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
                cv2.imshow('Input', frame)

                if (cv2.waitKey(300) & 0xFF == ord('q')):
                    break

            else:
                break

        vid.release()
        cv2.destroyAllWindows()

    # detect the ball
    def DetectBall(self, frame,min_radius):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # filter only blue color & Filter it
        lowerBound = np.array([100,100,20], dtype = "uint8")
        upperBound = np.array([130,255,255], dtype = "uint8")
        blueMask = cv2.inRange(hsv_frame, lowerBound, upperBound)


        blueMask = cv2.erode(blueMask, None, iterations=2)
        blueMask = cv2.dilate(blueMask, None, iterations=2)

        im2, contours, hierarchy = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = (-1, -1)
        # only proceed if at least one contour was found
        if len(contours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(blueMask)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > min_radius:
                # outline ball
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                # show ball center
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

        return center[0], center[1], radius


def main():
    processImg = ProcessVideo()
    processImg.DetectObject()


if __name__ == "__main__":
    main()

print('Program Completed!')
