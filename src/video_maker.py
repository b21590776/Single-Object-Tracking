import cv2
import os

image_folder = '../dataset/videos/train/video0001'
video_name = 'train.avi'

images =  [image for image in os.listdir(image_folder)]
first_frame = cv2.imread(os.path.join(image_folder, images[0]))

height, width, layers = first_frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()