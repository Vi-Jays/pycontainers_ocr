import platform
import pytesseract
import cv2 as cv

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = "D:\\workspaces\\dotpy\\pycontainers_ocr\\MainApp\\utils\\main\\ng_tesseract\\tesseract.exe"


def error_check(code_f):
    code_fx = list(code_f)
    n = len(code_fx)

    if n != 11:
        return 'UNKNOWN'
    else:
        for i in range(4):
            if code_fx[i] == '1':
                code_fx[i] = 'I'
            if code_fx[i] == '0':
                code_fx[i] = 'O'
            if code_fx[i] == '4':
                code_fx[i] = 'A'
            if code_fx[i] == '6':
                code_fx[i] = 'G'
            if code_fx[i] == '8':
                code_fx[i] = 'B'

        for i in range(4, 11):
            if code_fx[i] == 'I':
                code_fx[i] = '1'
            if code_fx[i] == 'O':
                code_fx[i] = '0'
            if code_fx[i] == 'A':
                code_fx[i] = '4'
            if code_fx[i] == 'G':
                code_fx[i] = '6'
            if code_fx[i] == 'B':
                code_fx[i] = '8'

    code_fx = ''.join(code_fx)
    return code_fx


def reformat_code(code):
    formatted = None
    n = len(code)

    if n <= 12:
        formatted = code[0:n-1]
        formatted = formatted.replace(' ', '')
        formatted = formatted.replace('\n', '')

    return formatted


def build_tesseract(is_sidecode=False):
    alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-l eng"
    if is_sidecode:
        options += " --psm 5"
    else:
        options += " --psm 12"
    options += " --oem 1"
    options += " -c tessedit_char_whitelist={}".format(alphanum)
    # Tesseract won't use dictionary
    options += " load_freq_dawg=false load_system_dawg=false"

    return options


def find_code(img):
    options = build_tesseract()

    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgray = cv.bilateralFilter(imgray, 11, 17, 17)

    rs = pytesseract.image_to_string(imgray, config=options)

    rs = reformat_code(rs)

    rs = error_check(rs)

    return rs
