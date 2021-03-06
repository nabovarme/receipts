import time
import os
import subprocess
from PIL import Image
import logging
import requests
from collections import namedtuple

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
PANIC = 8
CRASHED = 9
UNLOCK_PHONE = 10
BLACK_SCREEN_OF_DEATH = 11

HUMAN_LOOKUP = {
    OVERVIEW_LIST:'OVERVIEW_LIST',
    HOME:'HOME',
    LOGIN:'LOGIN',
    SEND_REQUEST_AND_PAY:'SEND_REQUEST_AND_PAY',
    MENU:'MENU',
    RECEIPT:'RECEIPT',
    EMPTY:'EMPTY',
    LOGOUT:'LOGOUT',
    PANIC:'PANIC',
    CRASHED: 'CRASHED',
    UNLOCK_PHONE: 'UNLOCK_PHONE',
    BLACK_SCREEN_OF_DEATH: 'BLACK_SCREEN_OF_DEATH'
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
        (192, 107), 124,
        (400, 118), 57,
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
    ],
    [
        (41, 355), 40,
        (41, 460), 40,
        (256, 460), 40,
        CRASHED
    ],
    [
        (425, 710), 40,
        (430, 742), 255,
        (416, 762), 41,
        UNLOCK_PHONE 
    ],
    [
        (425, 710), 0,
        (430, 742), 0,
        (416, 762), 0,
        BLACK_SCREEN_OF_DEATH 
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
        coords.append((pt1, pt2, pt3, state))
    if log_coordinates or not possible_states or HOME in possible_states :
        for coord in coords:
            print(HUMAN_LOOKUP[coord[3]], coord[:3], flush=True)
    if not possible_states:
        image.save('/images/unseen/{}.png'.format(int(time.time())))
        return PANIC
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
    cmd('shell input swipe 242 730 242 480')

def should_swipe_further_overview_list():  
    r = requests.get('http://ocr:8000/should_i_swipe_further')
    data = r.json()
    print(data, flush=True)
    receipts = data['receipts']
    return receipts, data['last_receipt']

def check_if_has_seen_first_receipt():  
    r = requests.get('http://ocr:8000/has_seen_first_receipt')
    data = r.json()
    seen = data['seen']
    return seen

def open_receipt(receipt):
    cmd("shell input tap {} {}".format(receipt['x'], receipt['y']))
    image = screenshot('receipt.png')
    return image

def mark_receipt_as_seen(receipt):
    r = requests.get('http://ocr:8000/see_receipt', json=receipt)

def close_receipt():
    cmd('shell input keyevent KEYCODE_BACK')

def reset_overview():
    cmd('shell input tap 42 77')
    open_activities()

def open_activities_from_send_payment():
    cmd('shell input tap 400 120')
    print("opening activities")

def empty_go_to_logout():
    cmd('shell input keyevent KEYCODE_BACK')
    cmd('shell input keyevent KEYCODE_BACK')

def perform_logout():
    cmd('shell input tap 356 745')

def go_back():
    cmd('shell input keyevent KEYCODE_BACK')

def ok_to_crash():
    cmd('shell input tap 252 467')

def unlock_phone():
    cmd('shell input swipe 23 426 444 426')

def enter_home_button():
    cmd('shell input keyevent KEYCODE_HOME')

def loop():
    STATE_OF_PANIC = False
    LOWEST_ROW = None
    LAST_STATE = None
    try:
        while True:
            image = screenshot()
            state = get_state(image)
            logging.warning("current state: "+HUMAN_LOOKUP[state])

            if state == HOME:
                logging.warning("ACTION: HOME")
                open_app()
                STATE_OF_PANIC = False
                LOWEST_ROW = None
            
            if state == LOGIN and LAST_STATE == LOGIN:
                logging.warning("ACTION: LOGIN and last state LOGIN")
                go_back()

            if state == LOGIN:
                logging.warning("ACTION: LOGIN")
                input_pin()

            if state == PANIC:
                logging.warning("ACTION: PANIC")
                STATE_OF_PANIC = True
            
            if state == RECEIPT:
                logging.warning("ACTION: RECEIPT")
                reset_overview()
            
            if state == UNLOCK_PHONE:
                logging.warning("ACTION: UNLOCK_PHONE")
                unlock_phone()
            
            if state == CRASHED:
                logging.warning("ACTION: CRASHED")
                ok_to_crash()
            
            if state == BLACK_SCREEN_OF_DEATH:
                logging.warning("ACTION: BLACK_SCREEN_OF_DEATH")
                enter_home_button()
            
            if state == SEND_REQUEST_AND_PAY or STATE_OF_PANIC:
                logging.warning("ACTION: SEND_REQUEST_AND_PAY or STATE_OF_PANIC")
                open_activities_from_send_payment()
            
            if state == EMPTY or STATE_OF_PANIC:
                logging.warning("ACTION: EMPTY or STATE_OF_PANIC")
                empty_go_to_logout()
            
            if state == LOGOUT:
                logging.warning("ACTION: LOGOUT")
                perform_logout()

            if state == OVERVIEW_LIST:
                logging.warning("ACTION: OVERVIEW_LIST")
                receipts, last_receipt = should_swipe_further_overview_list()
                for receipt in receipts:
                    image = open_receipt(receipt)
                    state = get_state(image)
                    if state != RECEIPT:
                        STATE_OF_PANIC = True
                        break
                    mark_receipt_as_seen(receipt)
                    close_receipt()
                    image = screenshot()
                    state = get_state(image)
                    if state != OVERVIEW_LIST:
                        break
                seen_first_receipt = check_if_has_seen_first_receipt()
                print("got", len(receipts), "and seen first receipt:", seen_first_receipt)
                if receipts or not seen_first_receipt:

                    swipe_down_overview_list()
                    if LOWEST_ROW is not None and LOWEST_ROW == last_receipt:
                        STATE_OF_PANIC = True
                    LOWEST_ROW = last_receipt
                else:
                    reset_overview()
            
            if state == MENU:
                logging.warning("ACTION: MENU")
                open_activities()
            LAST_STATE = state
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
