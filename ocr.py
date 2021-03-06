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

import pymysql.cursors
import pymysql
import imagehash

OverviewReceipt = namedtuple('OverviewReceipt', 'phash filename full_text')
Receipt = namedtuple('Receipt', 'filename full_text')


dbconfig = {
  "host": os.environ['DB_HOST'],
  "db": os.environ['DB_DATABASE'],
  "user":     os.environ['DB_USER'],
  "password": os.environ['DB_PASSWORD']
}
    # Connect to the database
CONNECTION = pymysql.connect(
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor, **dbconfig)

def image_to_hash(filename):
    image = Image.open(filename)
    return str(imagehash.phash(image, hash_size=16))
    
  
def auto_crop_image(cv_image_gray, filename):
    height, width = cv_image_gray.shape
    cv2.imwrite('/images/tmp_thrash.png', cv_image_gray)   
    ret,thresh = cv2.threshold(cv_image_gray,200,255,0)
    thresh = 255 - thresh

    _,contours,_ = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x,y, x+w,y+h])

    boxes = np.asarray(boxes)
    left = max(np.min(boxes[:,0])-2, 0)
    top = max(np.min(boxes[:,1])-2, 0)
    right = min(np.max(boxes[:,2]) + 2, width)
    bottom = min(np.max(boxes[:,3]) + 2, height)
    cv_image_gray = cv_image_gray[top:bottom, left:right]
    cv2.imwrite(filename, cv_image_gray)

def blur_and_threshold_image(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,20,
                            param1=50,param2=30,minRadius=40,maxRadius=50)
    output = gray.copy()

    y_index = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            y_index = y + 90

            cv2.circle(output, (x, y), r+4, (255,255,255), -1)
   
    blured_img = cv2.medianBlur(output,1)
    _, img = cv2.threshold(blured_img,175,255,cv2.THRESH_BINARY)

    if y_index:
        upper = img[:y_index, :]
        downer = img[y_index:, :]
        h, w = downer.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(downer, mask, (0,0), 0)
        downer = cv2.bitwise_not(downer)
        img = np.concatenate((upper, downer), axis=0)
    cv2.imwrite(filename, img[50:, :])


def overview_image_to_rows(filename):
    img_rgb = cv2.imread(filename)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    width = img_gray.shape[1]
    template = cv2.imread('/images/template.png',0)

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
        if y < 320:
            # ignoring rows until this point, since they are half height
            continue
        tmp_image = img_gray[pt[1]-120:pt[1] - 10 + h, 115:width-30]
        tmp_filename = f'/images/tmp/{index}.png'
        auto_crop_image(tmp_image, tmp_filename)
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
    return Receipt(filename, text)
    
def row_receipt_from_filename(filename):
    text = tesseract_on_filename(filename)
    return OverviewReceipt(image_to_hash(filename), filename, text)

def image_to_blob(filename):
    with open(filename, 'rb') as f:
        photo = f.read()
    return photo


def insert_into_db(overview_receipt, detail_receipt):


    query = """
        INSERT INTO accounts_auto (info_row, info_detail, screenshot_row, screenshot_detail, info_row_phash)
        VALUES (%s, %s, %s, %s, '{}');
    """.format( overview_receipt.phash )

    screenshot_row = image_to_blob(overview_receipt.filename)
    screenshot_detail = image_to_blob(detail_receipt.filename)

    args = (overview_receipt.full_text, detail_receipt.full_text, screenshot_row, screenshot_detail)

    try:
        with CONNECTION.cursor() as cursor:
            # Create a new record
            cursor.execute(query, args)

        # connection is not autocommit by default. So you must commit to save
        # your changes.
        CONNECTION.commit()
        logging.warning("succesfully inserted row")
    except:
        logging.error("DB REPORTED DUPLICATE")
        query = """
            UPDATE accounts_auto SET duplicate_count = duplicate_count + 1 where info_row = '{0}' AND info_detail = '{1}';
        """.format(overview_receipt.full_text, detail_receipt.full_text)
        try:
            with CONNECTION.cursor() as cursor:
                # Create a new record
                cursor.execute(query)
                CONNECTION.commit()
                logging.warning("UPDATED duplicate_count plus 1")
        except:
            logging.exception("UPDATE DUPLICATE_COUNT ERROR")

def should_checkout_row_receipt(row_receipt):
    # select * from stuff where info_row == this
    query = """
        SELECT * FROM accounts_auto WHERE 
     
        BIT_COUNT(CAST(CONV((SUBSTRING(info_row_phash, 1, 16)), 16, 10) AS UNSIGNED) ^ CAST(CONV(SUBSTRING('{0}', 1, 16), 16, 10) AS UNSIGNED)) +
        BIT_COUNT(CAST(CONV((SUBSTRING(info_row_phash, 17, 16)), 16, 10) AS UNSIGNED) ^ CAST(CONV(SUBSTRING('{0}', 17, 16), 16, 10) AS UNSIGNED)) +
        BIT_COUNT(CAST(CONV((SUBSTRING(info_row_phash, 33, 16)), 16, 10) AS UNSIGNED) ^ CAST(CONV(SUBSTRING('{0}', 33, 16), 16, 10) AS UNSIGNED)) +
        BIT_COUNT(CAST(CONV((SUBSTRING(info_row_phash, 49, 16)), 16, 10) AS UNSIGNED) ^ CAST(CONV(SUBSTRING('{0}', 49, 16), 16, 10) AS UNSIGNED)) < 1;
    """.format(row_receipt.phash)
    shoult_investigate = True
    try:
        with CONNECTION.cursor() as cursor:
            # Create a new record
            cursor.execute(query)
            count = cursor.rowcount
            print("have i seen {} before?, {}".format(row_receipt.phash, count))
            if count:
                shoult_investigate = False

        # connection is not autocommit by default. So you must commit to save
        # your changes.
        CONNECTION.commit()
    except:
        logging.exception("MYSQL ERROR")
    return shoult_investigate

def has_seen_first_receipt():
    # select * from stuff where info_row == this
    query = """
        SELECT * from accounts_auto WHERE info_row = %s
    """
    args = ('Martin Leidesdorff\nYou received money 40,00\n16.02.2017\n\x0c',)
    seen = False
    try:
        with CONNECTION.cursor() as cursor:
            # Create a new record
            cursor.execute(query, args)
            count = cursor.rowcount
            if count:
                seen = True
            
        # connection is not autocommit by default. So you must commit to save
        # your changes.
        CONNECTION.commit()
    except:
        logging.exception("MYSQL ERROR")
    return seen

app = Sanic()

@app.route("/should_i_swipe_further")
async def test(request):
    filename = '/images/screencap.png'
    receipts_to_open = []
    images_and_x_y = overview_image_to_rows(filename)
    last_receipt = None
    for index, (punkt, tmp_filename) in enumerate(images_and_x_y):
        if punkt[1] < 300:
            logging.error("ignoring image with y value lower than 300")
            continue
        overview_receipt = row_receipt_from_filename(tmp_filename)
        last_receipt = overview_receipt
        if should_checkout_row_receipt(overview_receipt):
            receipts_to_open.append((punkt, overview_receipt._asdict()))
    receipts = [
        {
            'x':np.asscalar(punkt[0]), 
            'y':np.asscalar(punkt[1]) - 30, 
            **overview_receipt
        } for punkt, overview_receipt in receipts_to_open
    ]
    return json({
        'receipts': receipts,
        'last_receipt': last_receipt._asdict()
    })


@app.route("/see_receipt")
async def test_see(request):
    receipt = request.json
    overview_receipt = OverviewReceipt(*(receipt[k] for k in OverviewReceipt._fields))
    if should_checkout_row_receipt(overview_receipt):
        receipt_detail_filename = '/images/receipt.png'
        blur_and_threshold_image(receipt_detail_filename)
        receipt = receipt_from_filename(receipt_detail_filename)
        print('\nSEEN RECEIPT\n', overview_receipt,'\n', receipt, '\n', flush=True)
        insert_into_db(overview_receipt, receipt)
    else:
        logging.warning("ignoring already seen receipt: {}".format(overview_receipt.full_text))

    return json({'seen':True})


@app.route("/has_seen_first_receipt")
async def test(request):
    seen = has_seen_first_receipt()
    return json({'seen':seen})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
