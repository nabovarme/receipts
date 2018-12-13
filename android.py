import time
import os
import subprocess
from PIL import Image
import logging
import requests
from collections import namedtuple

OverviewReceipt = namedtuple('OverviewReceipt', 'name amount date')
Receipt = namedtuple('Receipt', 'amount tlf')

ADB_IP = '10.8.0.86'

def cmd(cmd_line):
    cmd_line = 'adb -H {} {}'.format(ADB_IP, cmd_line)
    cmd = cmd_line.split()
    subprocess.call(cmd)
    time.sleep(2)

OVERVIEW_LIST = 0
HOME = 1
LOGIN = 2
SEND_REQUEST_AND_PAY = 3
MENU = 4
RECEIPT = 5
EMPTY = 6
LOGOUT = 7

HUMAN_LOOKUP = {
    OVERVIEW_LIST:'OVERVIEW_LIST',
    HOME:'HOME',
    LOGIN:'LOGIN',
    SEND_REQUEST_AND_PAY:'SEND_REQUEST_AND_PAY',
    MENU:'MENU',
    RECEIPT:'RECEIPT',
    EMPTY:'EMPTY',
    LOGOUT:'LOGOUT'
}

states = [
    [
        (124, 86), 55,
        (170,87), 55,
        (189, 75), 55,
        OVERVIEW_LIST
    ],
    [
        (9,780), 43,
        (471, 780), 76,
        (9, 40), 34,
        HOME
    ],
    [
        (190, 450),215,
        (280, 450), 215,
        (101, 450), 255,
        LOGIN
    ],
    [
        (192, 107), 57,
        (400, 118), 255,
        (264, 104), 56,
        SEND_REQUEST_AND_PAY
    ],
    [
        (119, 269), 55,
        (169, 275), 57,
        (219, 276), 57,
        MENU
    ],
    [
        (110, 81), 58,
        (151, 86), 55,
        (190, 80), 55,
        RECEIPT
    ],
    [
        (380, 619), 55,
        (283, 508), 193,
        (239, 539), 234,
        EMPTY
    ],
    [
        (108, 304), 255,
        (456, 784), 55,
        (22, 768), 55,
        LOGOUT
    ]
]

def get_state(image, log_coordinates=False):
    possible_states = []
    coords = []
    for coord1, gray1, coord2, gray2, coord3, gray3, state in states:
        pt1 = image.getpixel(coord1)
        pt2 = image.getpixel(coord2)
        pt3 = image.getpixel(coord3)
        if pt1[0] == gray1 and pt2[0] == gray2 and pt3[0] == gray3:
            possible_states.append(state)
    if log_coordinates or not possible_states:
        print(possible_states, coords, flush=True)
    state = possible_states[-1]
    return state




def screenshot(filename='screencap.png'):
    cmd('shell screencap -p /sdcard/screencap.png')
    cmd('pull /sdcard/screencap.png /images/' + filename)
    image = Image.open('/images/' + filename)
    image = image.convert('LA')
    return image

def input_pin():
    cmd("shell input text '3'")
    cmd("shell input text '7'")
    cmd("shell input text '3'")
    cmd("shell input text '7'")

def open_app():
    cmd('shell input tap 88 756')

def open_activities():
    cmd('shell input tap 116 346')

def swipe_down_overview_list():
    cmd('shell input swipe 242 736 244 461')

def should_swipe_further_overview_list():  
    r = requests.get('http://ocr:8000/should_i_swipe_further')
    data = r.json()
    print(data, flush=True)
    receipts = data['receipts']
    return receipts

def see_receipt(receipt):
    cmd("shell input tap {} {}".format(receipt['x'], receipt['y']))
    screenshot('receipt.png')
    r = requests.get('http://ocr:8000/see_receipt', json=receipt)
    data = r.json()
    print(data, flush=True)
    cmd('shell input keyevent KEYCODE_BACK')

def reset_overview():
    cmd('shell input tap 42 77')
    open_activities()

def open_activities_from_send_payment():
    cmd('shell input tap 408 109')
    time.sleep(2)

def empty_go_to_logout():
    cmd('shell input keyevent KEYCODE_BACK')

def perform_logout():
    #press logout button
    cmd('shell input tap 356 745')



#cmd('disconnect')

#cmd('connect 10.0.1.71:5555')





def loop():
    try:
        while True:
            image = screenshot()
            state = get_state(image)
            logging.warning("current state: "+HUMAN_LOOKUP[state])

            if state == HOME:
                open_app()
                        
            if state == LOGIN:
                input_pin()
            
            if state == RECEIPT:
                reset_overview()
            
            if state == SEND_REQUEST_AND_PAY:
                open_activities_from_send_payment()
            
            if state == EMPTY:
                empty_go_to_logout()
            
            if state == LOGOUT:
                perform_logout()

            if state == OVERVIEW_LIST:
                receipts = should_swipe_further_overview_list()
                for receipt in receipts:
                    see_receipt(receipt)
                    image = screenshot()
                    state = get_state(image)
                    if state != OVERVIEW_LIST:
                        break
                if receipts:
                    swipe_down_overview_list()
                else:
                    reset_overview()
            
            if state == MENU:
                open_activities()
            
    except:
        logging.exception("ERROR")

def test():
    TEST_DIR = '/images/test/'
    for filename in os.listdir(TEST_DIR):
        asserted_state = filename.split('.')[0]
        image = Image.open(TEST_DIR+filename)
        image = image.convert('LA')
        state = get_state(image, log_coordinates=True)
        logging.warning("asserted_state: "+asserted_state + " current state: "+HUMAN_LOOKUP[state])
        assert asserted_state == HUMAN_LOOKUP[state]
    exit(0)

#test()
loop()
