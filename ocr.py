from PIL import Image
import subprocess
import cv2
import os
import numpy as np
from collections import namedtuple
import dateutil.parser
import logging
import shutil
from sanic import Sanic
from sanic.response import json

OverviewReceipt = namedtuple('OverviewReceipt', 'name amount date error full_text')
Receipt = namedtuple('Receipt', 'name phone_number message error, full_text')


def overview_image_to_rows(filename):
    img_rgb = cv2.imread(filename)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    width = img_gray.shape[1]
    template = cv2.imread('/images/template_row.png',0)

    w, h = template.shape[::-1]
    cv2.imwrite("/images/tmp_img_gray.jpg", img_gray)


    res = cv2.matchTemplate(img_gray, template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    images_and_x_y = []
    last_y = 0
    index = 0
    for pt in zip(*loc[::-1]):
        y = pt[1]
        if y < last_y + 30:
            continue
        if y < 300:
            # ignoring rows until this point, since they are half height
            continue
        tmp_image = img_gray[pt[1]-120:pt[1] - 10 + h, 110:width-30]
        tmp_filename = f'/images/tmp/{index}.png'
        cv2.imwrite(tmp_filename, tmp_image)
        images_and_x_y.append((pt, tmp_filename))
        last_y = y
        index += 1
    return images_and_x_y

def tesseract_on_filename(filename):
    cmd_line = f"tesseract {filename} - --psm 4 --oem 1 -l dan+eng"
    text = subprocess.check_output(cmd_line.split())
    return text.decode('utf-8')

def receipt_from_filename(filename):
    text = tesseract_on_filename(filename)
    name = ''
    phone_number = ''
    message = text
    error = None
    full_text = text.replace('/', ' ')
    return Receipt(name, phone_number, message, error, full_text)
    
def row_receipt_from_filename(filename):
    text = tesseract_on_filename(filename)
    name, amount, date = text.split('\n')[:3]
    error = None
    full_text = text.replace('/', ' ')
    return OverviewReceipt(name, amount, date, error, full_text)


app = Sanic()

seen = []

@app.route("/should_i_swipe_further")
async def test(request):
    filename = '/images/screencap.png'
    receipts_to_open = []
    images_and_x_y = overview_image_to_rows(filename)
    for index, (punkt, tmp_filename) in enumerate(images_and_x_y):
        if punkt[1] < 300:
            logging.error("ignoring image with y value lower than 300")
            continue
        overview_receipt = row_receipt_from_filename(tmp_filename)
        filename = f"/images/errors/{overview_receipt.full_text}_row.jpg"
        shutil.copy(tmp_filename, filename)

        if overview_receipt not in seen:
            receipts_to_open.append((punkt, overview_receipt._asdict()))
    receipts = [
        {
            'x':np.asscalar(punkt[0]), 
            'y':np.asscalar(punkt[1]) - 30, 
            **overview_receipt
        } for punkt, overview_receipt in receipts_to_open
    ]
    return json({
        'receipts': receipts
    })

@app.route("/see_receipt")
async def test(request):
    receipt = request.json
    overview_receipt = OverviewReceipt(*(receipt[k] for k in OverviewReceipt._fields))
    filename = '/images/receipt.png'
    receipt = receipt_from_filename(filename)
    print('\nSEEN RECEIPT\n', overview_receipt,'\n', receipt, '\n', flush=True)

    seen.append(overview_receipt)
    shutil.copy('/images/receipt.png', f'/images/errors/{overview_receipt.full_text}_{receipt.full_text}_receipt.png')
    return json({'seen':True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
