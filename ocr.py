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
from mysql.connector import MySQLConnection, Error

OverviewReceipt = namedtuple('OverviewReceipt', 'filename full_text')
Receipt = namedtuple('Receipt', 'filename full_text')


dbconfig = {
  "database": os.environ['DB_HOST'],
  "user":     os.environ['DB_USER'],
  "password": os.environ['DB_PASSWORD']
}
CONNECTION_POOL = mysql.connector.pooling.MySQLConnectionPool(
    pool_name='666',
    pool_size=2,
    pool_reset_session=True,
    **self.dbconfig
)

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
    return Receipt(filename, text)
    
def row_receipt_from_filename(filename):
    text = tesseract_on_filename(filename)
    return OverviewReceipt(filename, text)

def image_to_blob(filename):
    with open(filename, 'rb') as f:
        photo = f.read()
    return photo


def insert_into_db(info, row_receipt_filename, receipt_detail_filename):
    
    query = """
        INSERT INTO accounts_auto
        VALUES(info, screnshow_row, screenshot_detail)
        (%s, %s, %s)
    """

    screenshot_row = image_to_blob(row_receipt_filename)
    screenshot_detail = image_to_blob(receipt_detail_filename)

    args = (info, screenshot_row, screenshot_detail)
 
    with CONNECTION_POOL.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, args)

        
app = Sanic()


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

    receipt_detail_filename = '/images/receipt.png'
    receipt = receipt_from_filename(filename)
    print('\nSEEN RECEIPT\n', overview_receipt,'\n', receipt, '\n', flush=True)
    info = '\n'.join([overview_receipt.full_text, receipt.full_text])
    insert_into_db(info, overview_receipt.filename, receipt.filename)
    return json({'seen':True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
