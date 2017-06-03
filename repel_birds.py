import numpy as np
import cv2
import time
import pyaudio
import wave
from os.path import join

from camera import Camera, FILE_DIRECTORY

history = []


def get_scale(image):
    max_dimensions = (750, 1200)
    dims = image.shape
    if 0 in dims:
        raise ValueError('Cannot show image without dimensions')
    scale = min([
        float(max_dimensions[0]) / dims[0],
        float(max_dimensions[1]) / dims[1]])
    return scale


def show_image(image, text='Image', time=0, destroy=True):
    scale = get_scale(image)
    resized_image = image.copy()
    resized_image = cv2.resize(resized_image, None, fx=scale, fy=scale)
    cv2.imshow(text, resized_image)
    k = cv2.waitKey(time)
    if destroy:
        cv2.destroyAllWindows()
    return k


def add_history(movement, human, start):
    global history
    keep_in_memory = 5 * 60  # seconds
    delete_buffer = 1 * 60   # seconds
    history.append([time.time() - start, movement, human])
    if len(history) > 2:
        diff = history[-1][0] - history[0][0]
        if diff > keep_in_memory + delete_buffer:
            for i in range(len(history)):
                if history[-1] - history[i] < keep_in_memory:
                    history = history[i:]
                    break


def take_action(last_action):
    min_duration = 3
    wait_time = 4

    if time.time() - last_action < wait_time:
        return False

    global history

    for k in range(len(history)):
        i = len(history) - 1 - k
        if history[i][1] is False or history[i][2] is True:
            return False
        elif history[-1][0] - history[i][0] > min_duration:
            return True
    return False

def play_sound(path):
    f = wave.open(path)
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                channels = f.getnchannels(),
                rate = f.getframerate(),
                output = True)

    data = f.readframes(chunk)
    while data:
        stream.write(data)
        data = f.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()


path = join(FILE_DIRECTORY, "sound.wav")
chunk = 1024







bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000)
c = Camera(camera_type='opencv', camera_index=0, resolution=(1920, 1080))
kernel = np.ones((3, 3), np.uint8)
detection_threshold = 5000
human_threshold = 100000

start = time.time()
last_time = time.time() - 30
while True:
    movement = False
    human = False
    img = c.capture(cv_pre_frames=1)
    mask = cv2.inRange(bg_subtractor.apply(img), 5, 255)
    mask = cv2.erode(mask, kernel, 1)
    for i in range(5):
        mask = cv2.dilate(mask, kernel, 1)
    out, contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: cv2.contourArea(x))
    img_mod = img
    if len(contours) and cv2.contourArea(contours[-1]) > detection_threshold:
        hull = cv2.convexHull(contours[-1])
        hull_mask = np.ones(img.shape) * 0.8
        hull_mask = cv2.drawContours(hull_mask, [hull], -1, [1, 1, 1], -1)
        img_mod = np.array(img.astype(np.float32) * hull_mask, dtype=np.uint8)
        movement = True
        if cv2.contourArea(contours[-1]) > human_threshold:
            human = True

    add_history(movement, human, start)
    if len(history) > 2:
        if history[-1][1] != history[-2][1] or (
                history[-1][2] != history[-2][2]):
            print time.time() - history[-1][0], history[-1][1:]
    if take_action(last_time) is True:
        print 'ACTION!'
        last_time = time.time()
        play_sound(path)
    show_image(
        img_mod, time=1, destroy=False)

