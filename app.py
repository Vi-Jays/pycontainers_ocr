from PIL import Image
from utils.main import Yolov7ONNX

import streamlit as st
import numpy as np
import cv2 as cv
import pandas as pd
import datetime
import os
import csv

###
st.set_page_config(
    page_title='CONTAINERS OCR',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'About': "*CONTAINERS OCR* é um programa desenvolvido em python para efectuar o monitoramento automático de "
                 "containers por reconhecimento de seus códigos."
    }
)


@st.cache()
def im_resize(im, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = im.shape[:2]

    if width is None and height is None:
        return im

    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv.resize(im, dim, interpolation=inter)

    return resized


@st.cache(allow_output_mutation=True)
def load_image(buff):
    return np.array(Image.open(buff))


PMODEL = 'ai/cocr.onnx'
PAPP_IMG = 'im.png'
IM_PCSV = 'img_database.csv'
MON_PCSV = 'mon_database.csv'
###

st.sidebar.title('CONTAINERS OCR')
st.sidebar.subheader('Definições')
menu = st.sidebar.selectbox(
    'MENU', ('APP', 'Monitor', 'Imagem (OCR)'))

headers = ['Data', 'Container ID']
filenames = [IM_PCSV, MON_PCSV]

for file in filenames:
    if not os.path.exists(file):
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

if menu == 'APP':
    # APP - HOME
    st.title('APP')
    st.markdown('---')
    st.write('#### **Tudo pronto?**')
    st.warning(
        '**CONTAINERS OCR** é um programa desenvolvido em python para efectuar o reconhecimento de código de containers.')
    im_main = load_image(PAPP_IMG)
    st.image(im_main,
             caption='Imagem de containers ocr', use_column_width='always')

if menu == 'Imagem (OCR)':
    # APP - C-IMOCR

    hist = list()

    buf_image = st.sidebar.file_uploader(
        '', type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])

    st.sidebar.markdown('---')
    st.sidebar.subheader('Imagem')

    imframe = st.sidebar.image([])

    st.sidebar.markdown('---')
    st.sidebar.subheader('Ajuste (parâmetros)')

    min_score = st.sidebar.slider(
        'Grau de detecção', min_value=0.0, max_value=1.0, value=0.3)

    yolonnx = Yolov7ONNX(PMODEL, min_score)

    st.title('CONTAINERS OCR')

    st.markdown('---')

    st_info = st.info(
        'Carregue uma imagem usando o painel lateral, nas definições.')

    btn_i = st.button('Inicializar')

    col1, col2, col3 = st.columns([2, 2, 3], gap='small')

    with col1:
        st.write('### Detectado:')
        crop_img = st.image([])

    with col2:
        st.write('### Container ID:')
        txt_container_id = st.empty()

    with col3:
        st.write('### Histórico')
        df = pd.read_csv(IM_PCSV)
        tb_hist = st.dataframe(df, width=300)

    if buf_image is not None:
        image = load_image(buf_image)
        image = im_resize(image, width=640)
        imframe.image(image)
        st_info.success(
            'Agora já pode inicializar o reconhecimento de código do container!')

    if btn_i and buf_image is not None:
        st_info.warning('A processar...')

        container, crop, container_id = yolonnx(image)

        if container:
            container = cv.imread(container, cv.COLOR_BGR2RGB)
            crop = cv.imread(crop, cv.COLOR_BGR2RGB)
            container = im_resize(container, width=640)
            st_info.image(container, use_column_width='always')
            crop_img.image(crop)
            txt_container_id.write('#### ' + container_id)
            now = datetime.datetime.now()
            dt_now = now.strftime("%d/%m/%Y, %H:%M:%S")
            hist.append((dt_now, container_id))
            df = pd.DataFrame(hist, columns=['Data', 'Container ID'])
            df.to_csv(IM_PCSV, index=False, header=False, mode='a')
            df = pd.read_csv(IM_PCSV)
            tb_hist.dataframe(df, width=300)
        else:
            st_info.error('Tente novamente.')

if menu == 'Monitor':
    # APP - C-MORC
    hist = list()

    st.sidebar.markdown('---')
    st.sidebar.subheader('Câmera')
    options = st.sidebar.selectbox('Modo', ('Desabilitada', 'Habilitada'))
    st.sidebar.markdown('---')
    st.sidebar.subheader('Ajuste (parâmetros)')

    min_score = st.sidebar.slider(
        'Grau de detecção', min_value=0.0, max_value=1.0, value=0.3)

    yolonnx = Yolov7ONNX(PMODEL, min_score)

    st.title('CONTAINERS OCR')

    st.markdown('---')

    stFrame = st.info('Considere habilitar a câmera.')

    if options == 'Habilitada':
        stFrame.success('Tudo pronto!')
        monitor_start = st.button(
            'Inicializar', help='Iniciar o reconhecimento automático de containers')

        col1, col2, col3 = st.columns([2, 2, 3], gap='small')

        with col1:
            st.write('### Detectado:')
            crop_img = st.image([])

        with col2:
            st.write('### Container ID:')
            txt_container_id = st.empty()

        with col3:
            st.write('### Histórico')
            df = pd.read_csv(MON_PCSV)
            tb_hist = st.dataframe(df, width=300)

        if monitor_start:
            stFrame.warning('A inicializar... aguarde')
            cap = cv.VideoCapture(0)
            startt = 0
            cap.set(cv.CAP_PROP_POS_FRAMES, startt * 60)

            while cap.isOpened():
                ret, frm = cap.read()
                if not ret:
                    stFrame.error('Câmera indisponível')
                    break

                container, crop, container_id = yolonnx(frm)
                if container:
                    crop = cv.imread(crop, cv.COLOR_BGR2RGB)
                    crop_img.image(crop)
                    txt_container_id.write('#### ' + container_id)
                    now = datetime.datetime.now()
                    dt_now = now.strftime("%d/%m/%Y, %H:%M:%S")
                    hist.append((dt_now, container_id))
                    df = pd.DataFrame(hist, columns=['Data', 'Container ID'])
                    df.to_csv(MON_PCSV, index=False, header=False, mode='a')
                    df = pd.read_csv(MON_PCSV)
                    tb_hist.dataframe(df, width=300)

                    container = cv.imread(container)
                    container = im_resize(container, width=640)

                    stFrame.image(
                        container, use_column_width='always', channels='BGR')
                else:
                    frm = im_resize(frm, width=640)
                    stFrame.image(frm, use_column_width='always',
                                  channels='BGR')
