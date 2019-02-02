

# -*- coding:utf-8 -*-
import cv2
import numpy as np

#cal the weight
def likelihood(x, y, func, image, w=30, h=30):
    x1 = np.int32(max(0, x - w / 2))
    y1 = np.int32(max(0, y - h / 2))
    x2 = np.int32(min(image.shape[1], x + w / 2))
    y2 = np.int32(min(image.shape[0], y + h / 2))
    #cut the area
    region = image[y1:y2, x1:x2]
    #count the pixel
    count = region[func(region)].size
    return (float(count) / image.size) if count > 0 else 0.0001


def init_particles(func, image):
    mask = image.copy()
    mask[func(mask) == False] = 0
    #detect the contour
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 0:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    max_rect = np.array(cv2.boundingRect(max_contour))
    max_rect = max_rect[:2] + max_rect[2:] / 2
    weight = likelihood(max_rect[0], max_rect[1], func, image)
    particles = np.ndarray((500, 3), dtype=np.float32)
    particles[:] = [max_rect[0], max_rect[1], weight]
    
    return particles

#resample according to the particles weight
def resample(particles):
    tmp_particles = particles.copy()
    weights = particles[:, 2].cumsum()
    last_weight = weights[weights.shape[0] - 1]
    for i in range(particles.shape[0]):
        weight = np.random.rand() * last_weight
        particles[i] = tmp_particles[(weights > weight).argmax()]
        particles[i][2] = 1.0

def predict(particles, variance=13.0):

    particles[:, 0] += np.random.randn((particles.shape[0])) * variance
    particles[:, 1] += np.random.randn((particles.shape[0])) * variance


def weight(particles, func, image):
    for i in range(particles.shape[0]):
        particles[i][2] = likelihood(particles[i][0], particles[i][1], func, image)
        #adding the weight
        sum_weight = particles[:, 2].sum()
        particles[:, 2] *= (particles.shape[0] / sum_weight)

def measure(particles):
    x = (particles[:, 0] * particles[:, 2]).sum()
    y = (particles[:, 1] * particles[:, 2]).sum()
    weight = particles[:, 2].sum()
    return x / weight, y / weight

#get the cords of particle
particle_filter_cur_frame = 0
def particle_filter(particles, func, image, max_frame=10):
    global particle_filter_cur_frame

    if image[func(image)].size <= 0:
        if particle_filter_cur_frame >= max_frame:
            return None, -1, -1
        particle_filter_cur_frame = min(particle_filter_cur_frame + 1, max_frame)
    else:
        particle_filter_cur_frame = 0
        if particles is None:
            particles = init_particles(func, image)

    if particles is None:
        return None, -1, -1

    resample(particles)
    predict(particles)
    weight(particles, func, image)
    x, y = measure(particles)
    return particles, x, y

if __name__ == '__main__':
    def is_color(region):
        #limit the blur color range
        return (region >= 90) & (region < 130)

    #input the vedio
    cap = cv2.VideoCapture('/Users/ph/Desktop/ball6.mp4')
    particles = None

    while cv2.waitKey(20) < 1:
        _, frame = cap.read()
        # Transfer to HSV format
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_h = frame_hsv[:, :, 0]
        _, frame_s = cv2.threshold(frame_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, frame_v = cv2.threshold(frame_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        frame_h[(frame_s == 0) | (frame_v == 0)] = 0


        particles, x, y = particle_filter(particles,is_color, frame_h)
        
        if particles is not None:
            valid_particles = np.int32(particles[(particles[:, 0] >= 0) & (particles[:, 0] < frame.shape[1]) &
                                        (particles[:, 1] >= 0) & (particles[:, 1] < frame.shape[0])])
            #modify the color of particles
            for i in range(valid_particles.shape[0]):
                frame[valid_particles[i][1], valid_particles[i][0]] = [0, 255, 255]
            p = np.array([x, y], dtype=np.int32)
    
            cv2.rectangle(frame, tuple(p - 15), tuple(p + 15), (0, 0, 255), thickness=2)


        cv2.imshow('green', frame)

    cap.release()
    cv2.destroyAllWindows()
