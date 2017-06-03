import cv2

from camera import Camera


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


c = Camera(camera_type='opencv', camera_index=1, resolution=(1920, 1080))
img = c.capture()
show_image(img)
