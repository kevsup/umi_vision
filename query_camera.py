from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
from blinkpy.helpers.util import json_load
import threading
import time

blink = Blink()
blink.auth = Auth(json_load("credentials.json"))
blink.start()

camera = blink.cameras["Living room"]
counter = 0

def capture():
    try:
        camera.snap_picture()
        time.sleep(10)
        blink.refresh()
        global counter
        camera.image_to_file('photos/' + 'umi_' + str(counter) + '.jpg')
        counter += 1
    except:
        pass
    threading.Timer(20.0, capture).start()

capture()
