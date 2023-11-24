"""
    msui.tutorials.tutorial_kml
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    This python script generates an automatic demonstration of how to overlay kml flles on top of the map in topview.
    kml(key hole markup language) is an XML based file format for demonstrating geographical context. This will
    demonstrate how to customize the kml files and other various operations on it.
    This file is part of MSS.

    :copyright: Copyright 2021 Hrithik Kumar Verma
    :copyright: Copyright 2021-2023 by the MSS team, see AUTHORS.
    :license: APACHE-2.0, see LICENSE for details.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import pyautogui as pag
import os.path

from pyscreeze import ImageNotFoundException
from tutorials.utils import platform_keys, start, finish, create_tutorial_images
from tutorials.utils.picture import picture


CTRL, ENTER, WIN, ALT = platform_keys()

def click_center_on_screen(pic, duration=2):
    x, y = pag.locateCenterOnScreen(pic)
    pag.click(x, y, duration=duration)


def select_listelement(steps):
    pag.press('down', presses=steps, interval=0.5)
    pag.press(ENTER, interval=1)
    pag.sleep(5)

def find_and_click_picture(pic_name, exception_message, duration=2):
    try:
        click_center_on_screen(picture(pic_name), duration)
        pag.sleep(1)
    except (ImageNotFoundException, OSError, Exception):
        print(f"\nException: {exception_message}")
        raise


def load_kml_file(pic_name, file_path, exception_message):
    try:
        find_and_click_picture(pic_name, exception_message)
        pag.typewrite(file_path, interval=0.1)
        pag.sleep(1)
        pag.press(ENTER)
    except (ImageNotFoundException, OSError, Exception):
        print(exception_message)
        raise


def change_attribute(pic_name, exception_message, actions, interval=2, sleep_time=2):
    try:
        click_center_on_screen(picture(pic_name), interval)
        pag.sleep(sleep_time)
        actions()
    except (ImageNotFoundException, OSError, Exception):
        print(f"\nException: {exception_message}")
        raise


def automate_kml():
    pag.sleep(5)
    path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    kml_file_path1 = os.path.join(path, 'docs/samples/kml/folder.kml')
    kml_file_path2 = os.path.join(path, 'docs/samples/kml/color.kml')

    hotkey = WIN, 'pageup'
    try:
        pag.hotkey(*hotkey)
    except Exception:
        print("\nException : Enable Shortcuts for your system or try again!")
    pag.hotkey('CTRL', 'h')
    pag.sleep(1)
    create_tutorial_images()

    find_and_click_picture('topviewwindow-01-europe-cyl.png',
                           "Map change dropdown could not be located on the screen")
    select_listelement(2)
    find_and_click_picture('topviewwindow-select-to-open-control.png',
                           "'select to open control' button/option not found on the screen.")
    select_listelement(4)
    create_tutorial_images()

    load_kml_file('topviewwindow-add-kml-files.png', kml_file_path1,
                  "'Add KML Files' button not found on the screen.")
    load_kml_file('topviewwindow-add-kml-files.png', kml_file_path2,
                  "'Add KML Files' button not found on the screen.")

    find_and_click_picture('topviewwindow-unselect-all-files.png',
                           "'Select All Files(Unselecting & Selecting)' "
                           "button not found on the screen.")
    find_and_click_picture('topviewwindow-select-all-files.png',
                           "'Select All Files(Unselecting & Selecting)' "
                           "button not found on the screen.")

    create_tutorial_images()

    pag.move(-200, 0, duration=1)
    pag.click(interval=2)

    # ToDo color is clicked but did not change, QT bug?
    change_attribute('topviewwindow-change-color.png',
                     "'Change Color' button not found on the screen.",
                     lambda: (pag.move(-220, -300, duration=1),
                              pag.click(interval=2),
                              pag.press(ENTER)),
                     interval=2)

    change_attribute('topviewwindow-2-00.png',
                     "'Change Linewidth' button not found on the screen.",
                     lambda: (pag.hotkey(CTRL,  'a'),
                              [pag.press('down') for _ in range(8)],
                              pag.hotkey(CTRL, 'a'),
                              pag.typewrite('2.50', interval=1),
                              pag.press(ENTER),
                              pag.sleep(1),
                              pag.hotkey(CTRL, 'a'),
                              pag.typewrite('4.50', interval=1),
                              pag.press(ENTER)),
                     interval=2)
    print("\nAutomation is over for this tutorial. Watch next tutorial for other functions.")
    finish()


if __name__ == '__main__':
    start(target=automate_kml, duration=220)
