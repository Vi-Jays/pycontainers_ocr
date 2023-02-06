import numpy as np
import cv2 as cv
import os


def letterbox(im0, nw_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im0.shape[:2]

    if isinstance(nw_shape, int):
        nw_shape = (nw_shape, nw_shape)

    r = min(nw_shape[0] / shape[0], nw_shape[1] / shape[1])

    if not scaleup:
        r = min(r, 1.0)

    nw_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = nw_shape[1] - nw_unpad[0], nw_shape[0] - nw_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != nw_unpad:
        im0 = cv.resize(im0, nw_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im0 = cv.copyMakeBorder(im0, top, bottom, left, right,
                            cv.BORDER_CONSTANT, value=color)  # add border
    return im0, r, (dw, dh)


def crop(x0, y0, x1, y1, frm_r_img):
    frm_r_img = frm_r_img[y0:y1+1, x0:x1+1]
    
    return frm_r_img


def save_crop(cropped_frm_r_img):
    f = 'out/{}.png'.format(os.getpid())
    cv.imwrite(f, cropped_frm_r_img)

    return f

def save_container(final_frm_r_img, text):
    f = 'containers/container_{}.png'.format(text)
    cv.imwrite(f, final_frm_r_img)

    return f
