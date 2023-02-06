from time import perf_counter
from .ocr import find_code
from .utils import crop, letterbox, save_container, save_crop

import numpy as np
import cv2 as cv
import onnxruntime as ort


class Yolov7ONNX:

    def __init__(self, model, conf=0.3):
        self.confidence_thres = conf
        self.class_names = ['Container Codes']
        self.colors = {names: [np.random.randint(0, 255) for _ in range(
            3)] for i, names in enumerate(self.class_names)}

        self.innit_model(model)

    def __call__(self, inpt_frm_r_img):
        inpt_frm_r_img = cv.cvtColor(inpt_frm_r_img, cv.COLOR_BGR2RGB)

        self.frm_r_img = inpt_frm_r_img

        imcopy = self.frm_r_img.copy()

        im_prep, ratio, dwdh = self.preprocess_frm_r_img(imcopy)

        return self.detect_n_recon_obj(im_prep, ratio, dwdh)

    def innit_model(self, path, gpu=False):
        if gpu:
            prvder = ['CUDAExecutionProvider']
        else:
            prvder = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(path, providers=prvder)

        self.out_name = [i.name for i in self.session.get_outputs()]
        
        self.inpt_name = [i.name for i in self.session.get_inputs()]
        

    def preprocess_frm_r_img(self, imcopy):
        im, ratio, dwdh = letterbox(imcopy, auto=False)
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)

        im_prep = im.astype(np.float32)
        im_prep /= 255

        return im_prep, ratio, dwdh

    def postprocess_frm_r_img(self, outputs):
        im0 = [self.frm_r_img.copy()]
        postpn = False

        for i, (batch, x0, y0, x1, y1, cls_id, conf) in enumerate(outputs):
            
            if conf >= self.confidence_thres:
                im = im0[int(batch)]
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

                bbox = np.array([x0, y0, x1, y1])
                bbox -= np.array(self.dwdh * 2)
                bbox /= self.ratio
                bbox = bbox.round().astype(np.int32).tolist()

                cropd = crop(bbox[0], bbox[1], bbox[2], bbox[3], im)
                
                txt = find_code(cropd)

                if txt is not None and txt != 'UNKNOWN':
                    cls_id = int(cls_id)

                    name = self.class_names[cls_id]
                    color = self.colors[name]

                    cv.rectangle(im, bbox[:2], bbox[2:], color, 2)
                    # cv.putText(
                    #    im, txt, (bbox[0], bbox[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.75, [0, 0, 0], 3)
                    postpn = True
        if postpn:
            return im, cropd, txt
        else:
            return False

    def detect_n_recon_obj(self, im_prep, ratio, dwdh):
        outputs = self.onnxrun(im_prep)

        self.ratio = ratio
        self.dwdh = dwdh

        if self.postprocess_frm_r_img(outputs):
            im, cropd, txt = self.postprocess_frm_r_img(outputs)

            sv_crop = save_crop(cropd)

            sv_container = save_container(im, txt)

            return sv_container, sv_crop, txt
        else:
            return False, False, False

    def onnxrun(self, im_prep):

        inpt = {self.inpt_name[0]: im_prep}

        startime = perf_counter()

        outs = self.session.run(self.out_name, inpt)[0]

        endtime = perf_counter()

        print('Inference: ({:.2f})s'.format(endtime - startime))
        # print(outs)

        return outs
