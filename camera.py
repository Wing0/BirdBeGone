import cv2
import os
import datetime
import oskui
import numpy as np

FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


class Camera(object):
    """
    Camera object for taking pictures with wide range of cameras.
    """

    camera = None

    def __init__(
            self, name=None, camera_type=None,
            camera_index=None, resolution=None):
        super(Camera, self).__init__()
        self.name = name
        self.camera_type = camera_type
        self.resolution = resolution

        self._init_camera(self.camera_type, camera_index)
        self._set_resolution(self.resolution)

    def _init_camera(self, camera_type, camera_index):
        if self.camera_type == 'basler':
            import pypylon
            devices = pypylon.factory.find_devices()
            if len(devices) == 0:
                print (
                    'Did you run:\n'
                    'export LD_LIBRARY_PATH=/Library/Frameworks/'
                    'pylon.framework/Libraries'
                )
                return False
            self.camera = pypylon.factory.create_device(devices[camera_index])
            self.camera.open()
            self.camera.properties['PixelFormat'] = 'RGB8'

        elif self.camera_type == 'opencv':
            self.camera = cv2.VideoCapture(camera_index)

        elif self.camera_type == 'picamera':
            import picamera
            from picamera.array import PiRGBArray
            self.camera = picamera.PiCamera()
            self.rgbarray = PiRGBArray(self.camera, format='bgr')

    def _set_resolution(self, resolution):
        if self.camera_type == 'basler':
            self.camera.properties['Width'] = resolution[0]
            self.camera.properties['Height'] = resolution[1]
        elif self.camera_type == 'opencv':
            self.camera.set(
                cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        elif self.camera_type == 'picamera':
            import picamera
            self.camera = picamera.PiCamera(
                resolution=resolution,
                framerate=30)

    def _set_gain(self, gain):
        '''
        Set the gain/iso of the camera
        @params:
        - gain: float from 0 to 1 or 'auto', scales gain to available range
        '''
        if self.camera_type == 'basler':
            if gain == 'auto':
                self.camera.properties['GainAuto'] = 'Continuous'
            else:
                self.camera.properties['GainAuto'] = 'Off'
                self.camera.properties['Gain'] = gain * 24.0
        elif self.camera_type == 'opencv':
            if gain == 'auto':
                self.camera.set(cv2.CAP_PROP_GAIN, 0)
            else:
                self.camera.set(cv2.CAP_PROP_GAIN, gain)
        elif self.camera_type == 'picamera':
            if gain == 'auto':
                self.camera.iso = 100
            else:
                self.camera.iso = 100 + gain * 700.0

    def _set_exposure(self, exposure_time):
        '''
        Set the exposure time of the camera. May not work properly on OpenCV
        Basler and picamera fully supported.
        @params:
        - exposure_time: float or 'auto', exposure time/shutter speed in ms
        '''
        if self.camera_type == 'basler':
            if exposure_time == 'auto':
                self.camera.properties['ExposureMode'] = 'Timed'
                self.camera.properties['ExposureAuto'] = 'Continuous'
            else:
                self.camera.properties['ExposureMode'] = 'Timed'
                self.camera.properties['ExposureAuto'] = 'Off'
                self.camera.properties['ExposureTime'] = exposure_time
        elif self.camera_type == 'opencv':
            if exposure_time == 'auto':
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            else:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_time / 1000.0)
        elif self.camera_type == 'picamera':
            if exposure_time == 'auto':
                self.camera.exposure_mode = 'auto'
                self.camera.shutter_speed = 0
            else:
                self.camera.exposure_mode = 'off'
                self.camera.shutter_speed = exposure_time * 1000

    def _set_scale(self, scale):
        binning = int(1.0 / scale)
        if self.camera_type == 'basler':
            self.camera.properties['BinningVertical'] = binning
            self.camera.properties['BinningHorizontal'] = binning
        elif self.camera_type == 'picamera':
            self._set_resolution(
                (scale * self.resolution[0], scale * self.resolution[1]))
        # ToDo: openCV binning

    def capture(self, folder='img', name=False, undistort=False, exposure=None,
                gain=0, iso=0, properties=None, ask=False, resolution=None,
                save=None, cv_pre_frames=5, scale=1.0):
        '''
        Cross platform and camera compatible image capture function
        @params:
        - exposure: float, exposure time in milliseconds, max 1000.0
        '''

        if ask:
            path = oskui.ask_file()
            img = cv2.imread(path)
            if save:
                return img, path
            else:
                return img

        if exposure is not None:
            self._set_exposure(exposure)

        if gain is not None or iso is not None:
            self._set_gain(gain if not iso else iso)

        if resolution is not None:
            self._set_resolution(resolution)

        self._set_scale(scale)

        if self.camera_type == 'picamera':
            img = self.camera.capture(self.rgbarray)
        elif self.camera_type == 'opencv':
            for i in range(cv_pre_frames):
                ret, img = self.camera.read()
        elif self.camera_type == 'basler':
            images = self.camera.grab_images(1)
            pre = [o for o in images][0]

            img = np.zeros((pre.shape[0], pre.shape[1] / 3, 3), dtype=np.uint8)
            for j in range(pre.shape[1]):
                c = int(j / 3)
                img[:, c, 2 - (j % 3)] = pre[:, j]

        if resolution is not None:
            self._set_resolution(self.resolution)

        if scale != 1.0:
            self._set_scale(1.0)

        if undistort:
            img = self.undistort(img)

        if save:
            if folder[-1] == '/':
                folder = folder[:-1]
            folder = FILE_DIRECTORY + '/' + folder
            if not os.path.exists(folder):
                os.makedirs(folder)
            name = name if name else str(
                datetime.datetime.now()).replace('.', '_')
            path = '%s/%s.jpg' % (folder, name)

            cv2.imwrite(path, img)
            return img, path

        return img

    def shutdown(self):
        if self.camera_type == 'basler':
            self.camera.close()
        elif self.camera_type == 'opencv':
            self.camera.release()

        cv2.destroyAllWindows()

    def undistort(self, img, alpha=1):
        '''
        Undistorts the image.
        Attention:
        - works only on resolution used in calibration
        - not to be used in pose estimation, the distortions
            are already taken into account
        '''
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), alpha, (w, h))
        dst = cv2.undistort(
            img, self.mtx, self.dist, None, newcameramtx)

        return dst


def discover_cameras(max_cameras=10):
    result = {'opencv': 0, 'picamera': 0, 'basler': 0}

    for number in range(max_cameras):
        camera = cv2.VideoCapture(number)
        if camera.isOpened():
            result['opencv'] += 1
        camera.release()
    try:
        import pypylon
        result['basler'] = len(pypylon.factory.find_devices())
    except Exception:
        pass

    try:
        import picamera
        camera = picamera.PiCamera()
        result['picamera'] = 1
    except Exception:
        pass

    return result


def find_exposure(camera, threshold=0.01, white=True, margin=5):
    scale = 0.25

    e = 100000 * white + (1 - white) * 10
    g = 0
    step = 0.1
    bleed = 1

    while bleed > threshold:
        if white:
            e *= step
        else:
            e /= step

        if e < 10:
            e = 10
            break
        img = camera.capture(exposure=e, gain=g, scale=scale)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_ar = img.flatten()
        if white:
            bleed = len(
                np.where(img_ar > 255 - margin)[0]) / float(len(img_ar))
        else:
            bleed = len(np.where(img_ar < margin)[0]) / float(len(img_ar))

    bleed = 0
    step = 1.1
    adjusted = False
    while bleed < threshold:
        if white:
            e *= step
        else:
            e /= step
        if e > 1000000:
            e = 1000000
            break
        adjusted = True
        img = camera.capture(exposure=e, gain=g, scale=scale)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_ar = img.flatten()
        if white:
            bleed = len(
                np.where(img_ar > 255 - margin)[0]) / float(len(img_ar))
        else:
            bleed = len(np.where(img_ar < margin)[0]) / float(len(img_ar))

    # Return step the step that was not above threshold
    if adjusted:
        if white:
            e /= step
        else:
            e *= step
    return e
