import cv2
import matplotlib.pyplot as plt


def plot(im, iris, eye_contour, plot_eye = True, plot_pupil = True):
    im = pad_image(im, desired_size=64)

    lm = iris[0]
    h, w, _ = im.shape
    if plot_pupil:
        cv2.circle(im, (int(lm[0]), int(lm[1])), 1, (0, 255, 0), 1)
        cv2.circle(im, (int(lm[3]), int(lm[4])), 1, (255, 0, 255), 1)
        cv2.circle(im, (int(lm[6]), int(lm[7])), 1, (255, 0, 255), 1)
        cv2.circle(im, (int(lm[9] ), int(lm[10])), 1, (255, 0, 255), 1)
        cv2.circle(im, (int(lm[12] ), int(lm[13])), 1, (255, 0, 255), 1)
    if plot_eye:
        for idx in range(71):
            cv2.circle(im, (int(eye_contour[0][idx*3]), int(eye_contour[0][idx*3 + 1])), 1, (0, 0, 255), -1)

    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(im)
    plt.show()


def pad_image(im, desired_size=64):
    
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    new_im.shape
    return new_im