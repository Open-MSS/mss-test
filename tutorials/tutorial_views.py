"""
    msui.tutorials.tutorial_views
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This python script generates an automatic demonstration of how to use the top view, side view, table view and
    linear view section of Mission Support System in creating a operation and planning the flightrack.

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

from sys import platform
from pyscreeze import ImageNotFoundException
from tutorials.utils import platform_keys, start, finish, create_tutorial_images, get_region
from tutorials.pictures import picture


# ToDo in sideview and topview waypoint movement needs adjustment

def automate_views():
    """
    This is the main automating script of the MSS views tutorial which will cover all the views(topview, sideview,
    tableview, linear view) in demonstrating how to create a operation. This will be recorded and savedto a file having
    dateframe nomenclature with a .mp4 extension(codec).
    """
    # Giving time for loading of the MSS GUI.
    pag.sleep(5)
    ctrl, enter, win, alt = platform_keys()

    # Screen Resolutions
    sc_width, sc_height = pag.size()[0] - 1, pag.size()[1] - 1

    # Maximizing the window
    try:
        if platform == 'linux' or platform == 'linux2':
            pag.hotkey('winleft', 'pageup')
        elif platform == 'darwin':
            pag.hotkey('ctrl', 'command', 'f')
        elif platform == 'win32':
            pag.hotkey('win', 'up')
    except Exception:
        print("\nException : Enable Shortcuts for your system or try again!")
        raise
    pag.sleep(2)
    pag.hotkey('ctrl', 'h')
    pag.sleep(2)
    create_tutorial_images()

    # Shifting topview window to upper right corner
    try:
        x, y = pag.locateCenterOnScreen(picture('topviewwindow-ins-wp.png'))
        pag.click(x, y - 56, interval=2)
        if platform == 'win32' or platform == 'darwin':
            pag.dragRel(525, -110, duration=2)
        elif platform == 'linux' or platform == 'linux2':
            pag.dragRel(910, -25, duration=2)
        pag.move(0, 56)
        add_tv_x, add_tv_y = pag.position()
        pag.move(-486, -56, duration=1)
        pag.click(interval=1)
        if platform == 'win32' or platform == 'linux' or platform == 'linux2':
            pag.hotkey('ctrl', 'v')
        elif platform == 'darwin':
            pag.hotkey('command', 'v')
        pag.sleep(4)
        create_tutorial_images()

        # Shifting Sideview window to upper left corner.
        try:
            x1, y1 = pag.locateCenterOnScreen(picture('sideviewwindow-ins-wp.png'))
            if platform == 'win32' or platform == 'darwin':
                pag.moveTo(x1, y1 - 56, duration=1)
                pag.dragRel(-494, -177, duration=2)
            elif platform == 'linux' or platform == 'linux2':
                pag.moveTo(x1, y1 - 56, duration=1)
                pag.dragRel(-50, -30, duration=2)
            pag.sleep(2)
            if platform == 'linux' or platform == 'linux2':
                pag.keyDown('altleft')
                # ToDo selection of views have to be done with ctrl f
                # this selects the next window in the window manager on budgie
                pag.press('tab')
                pag.keyUp('tab')
                pag.press('tab')
                pag.keyUp('tab')
                pag.keyUp('altleft')
            elif platform == 'win32':
                pag.keyDown('alt')
                pag.press('tab')
                pag.press('right')
                pag.keyUp('alt')
            elif platform == 'darwin':
                pag.press('command', 'tab', 'right')
            pag.sleep(1)
        except (ImageNotFoundException, OSError, Exception):
            print("Exception: \'Side View Window Header\' was not found on the screen")
            raise
    except (ImageNotFoundException, OSError, Exception):
        print("Exception: \'Topview Window Header\' was not found on the screen")
        raise

    # Adding waypoints
    if add_tv_x is not None and add_tv_y is not None:
        pag.sleep(1)
        pag.click(add_tv_x, add_tv_y, interval=2)
        pag.move(-50, 150, duration=1)
        pag.click(interval=2)
        pag.sleep(1)
        pag.move(65, 65, duration=1)
        pag.click(interval=2)
        pag.sleep(1)

        pag.move(-150, 30, duration=1)
        x1, y1 = pag.position()
        pag.click(interval=2)
        pag.sleep(1)
        pag.move(200, 150, duration=1)
        pag.click(interval=2)
        x2, y2 = pag.position()
        pag.sleep(1)
        pag.move(100, -80, duration=1)
        pag.click(interval=2)
        pag.move(56, -63, duration=1)
        pag.click(interval=2)
        pag.sleep(3)
    else:
        print("Screen coordinates not available for add waypoints for topview")
        raise

    # Locating Server Layer
    try:
        x, y = pag.locateCenterOnScreen(picture('topviewwindow-server-layer.png'),
                                        region=(int(sc_width / 2) - 100, 0, sc_width, sc_height))
        pag.click(x, y, interval=2)
        create_tutorial_images()
        # Entering wms URL
        try:
            x, y = pag.locateCenterOnScreen(picture('multilayersdialog-http-localhost-8081.png'),
                                            region=(int(sc_width / 2), 0, sc_width, sc_height))
            pag.click(x + 220, y, interval=2)
            pag.hotkey('ctrl', 'a', interval=1)
            pag.write('http://open-mss.org/', interval=0.25)
        except (ImageNotFoundException, OSError, Exception):
            print("\nException : Topviews' \'WMS URL\' editbox button/option not found on the screen.")
            raise

        create_tutorial_images()
        try:
            x, y = pag.locateCenterOnScreen(picture('multilayersdialog-get-capabilities.png'))
            pag.click(x, y, interval=2)
            pag.sleep(4)
        except (ImageNotFoundException, OSError, Exception):
            print("\nException : Topviews' \'Get capabilities\' button/option not found on the screen.")
            raise

        # Relocating Layerlist of topview
        if platform == 'win32':
            pag.move(-171, -390, duration=1)
            pag.dragRel(10, 627, duration=2)
        elif platform == 'linux' or platform == 'linux2' or platform == 'darwin':
            pag.move(-171, -390, duration=1)
            pag.dragRel(10, 675, duration=2)  # To be decided
        pag.sleep(1)
        # Storing screen coordinates for List layer of top view
        ll_tov_x, ll_tov_y = pag.position()
    except (ImageNotFoundException, OSError, Exception):
        print("\nException : Topviews WMS' \'Server\\Layers\' button/option not found on the screen.")
        pag.press('enter', interval=1)
        # raise

    # Selecting some layers in topview layerlist
    # lookup layer entry from the multilayering checkbox
    try:
        x, y = pag.locateCenterOnScreen(picture('multilayersdialog-multilayering.png'))
        # Divergence and Geopotential
        pag.click(x + 50, y + 70, interval=2)
        pag.sleep(1)
        # Relative Huminidity
        pag.click(x + 50, y + 110, interval=2)
        pag.sleep(1)

    except (ImageNotFoundException, OSError, Exception):
        print("\nException : \'Multilayering \' checkbox not found on the screen.")
        raise

    # Moving waypoints in Topview
    try:
        x, y = pag.locateCenterOnScreen(picture('topviewwindow-mv-wp.png'),
                                        region=(int(sc_width / 2) - 100, 0, sc_width, sc_height))
        pag.click(x, y, interval=2)
        # move point x1,y1
        pag.click(x1, y1, interval=2)
        pag.moveTo(x1, y1, duration=1)
        pag.dragTo(x1 + 46, y1 - 67, duration=1, button='left')
        pag.click(interval=2)
        x3, y3 = pag.position()
        pag.sleep(1)
    except ImageNotFoundException:
        print("\n Exception : Move Waypoint button could not be located on the screen")
        raise
    # Deleting waypoints
    try:
        x, y = pag.locateCenterOnScreen(picture('topviewwindow-del-wp.png'),
                                        region=(int(sc_width / 2) - 100, 0, sc_width, sc_height))
        pag.click(x, y, interval=2)
        pag.moveTo(x3, y3, duration=1)
        pag.click(duration=1)
        if platform == 'win32':
            pag.press('left')
        pag.sleep(2)
        if platform == 'linux' or platform == 'linux2' or platform == 'win32':
            pag.press('enter', interval=1)
        elif platform == 'darwin':
            pag.press('return', interval=1)
        pag.sleep(2)
    except ImageNotFoundException:
        print("\n Exception : Remove Waypoint button could not be located on the screen")
        raise

    # Changing map to Global
    try:
        if platform == 'linux' or platform == 'linux2' or platform == 'darwin':
            x, y = pag.locateCenterOnScreen(picture('topviewwindow-01-europe-cyl.png'),
                                            region=(int(sc_width / 2) - 100, 0, sc_width, sc_height))
            pag.click(x, y, interval=2)
        pag.press('down', presses=2, interval=0.5)
        if platform == 'linux' or platform == 'linux2' or platform == 'win32':
            pag.press('enter', interval=1)
        elif platform == 'darwin':
            pag.press('return', interval=1)
        pag.sleep(6)
    except (ImageNotFoundException, TypeError, OSError, Exception):
        print("\n Exception : Topview's Map change dropdown could not be located on the screen")
        raise

    # Zooming into the map
    try:
        x, y = pag.locateCenterOnScreen(picture('topviewwindow-zoom.png'),
                                        region=(int(sc_width / 2) - 100, 0, sc_width, sc_height))
        pag.click(x, y, interval=2)
        pag.move(155, 121, duration=1)
        pag.click(duration=1)
        pag.dragRel(260, 110, duration=2)
        pag.sleep(4)
    except ImageNotFoundException:
        print("\n Exception : Topview's Zoom button could not be located on the screen")
        raise
    # SideView Operations
    # Opening web map service

    try:
        x, y = pag.locateCenterOnScreen(picture('sideviewwindow-select-to-open-control.png'))
        pag.click(x, y, interval=2)
        pag.press('down', interval=1)
        if platform == 'linux' or platform == 'linux2' or platform == 'win32':
            pag.press('enter', interval=1)
        elif platform == 'darwin':
            pag.press('return', interval=1)
    except (ImageNotFoundException, OSError, Exception):
        print("\nException :\'SideView's select to open control\' button/option not found on the screen.")
        raise
    # Locating Server Layer
    try:
        x, y = pag.locateCenterOnScreen(picture('sideviewwindow-server-layer.png'),
                                        region=(0, 0, int(sc_width / 2) - 100, sc_height))
        pag.click(x, y, interval=2)
        # Entering wms URL
        try:
            x, y = pag.locateCenterOnScreen(picture('multilayersdialog-http-localhost-8081.png'))
            pag.click(x + 220, y, interval=2)
            pag.hotkey('ctrl', 'a', interval=1)
            pag.write('http://open-mss.org/', interval=0.25)
        except (ImageNotFoundException, OSError, Exception):
            print("\nException : Sideviews' \'WMS URL\' editbox button/option not found on the screen.")
            raise
        try:
            x, y = pag.locateCenterOnScreen(picture('multilayersdialog-get-capabilities.png'))
            pag.click(x, y, interval=2)
            pag.sleep(3)
        except (ImageNotFoundException, OSError, Exception):
            print("\nException : SideView's \'Get capabilities\' button/option not found on the screen.")
            pag.press('enter', interval=1)
            # raise

        if platform == 'win32':
            pag.move(-171, -390, duration=1)
            pag.dragRel(10, 570, duration=2)
        elif platform == 'linux' or platform == 'linux2' or platform == 'darwin':
            pag.move(-171, -390, duration=1)
            pag.dragRel(10, 600, duration=2)
        # Storing screen coordinates for List layer of side view
        ll_sv_x, ll_sv_y = pag.position()
        pag.sleep(1)
    except (ImageNotFoundException, OSError, Exception):
        print("\nException : Sideviews WMS' \'Server\\Layers\' button/option not found on the screen.")
        raise
    # Selecting some layers in Sideview WMS
    if platform == 'win32':
        gap = 22
    elif platform == 'linux' or platform == 'linux2' or platform == 'darwin':
        gap = 16

    try:
        x, y = pag.locateCenterOnScreen(picture('multilayersdialog-multilayering.png'))
        # Cloudcover
        pag.click(x + 50, y + 70, interval=2)
        pag.sleep(1)
        temp1, temp2 = x, y
        pag.click(x, y, interval=2)
        pag.sleep(3)
        pag.move(0, gap, duration=1)
        pag.click(interval=1)
        pag.sleep(3)
        pag.move(0, gap * 2, duration=1)
        pag.click(interval=1)
        pag.sleep(3)
        pag.move(0, gap, duration=1)
        pag.click(interval=1)
        pag.sleep(3)
        pag.move(0, -gap * 4, duration=1)
        pag.click(interval=1)
        pag.sleep(3)
    except (ImageNotFoundException, OSError, Exception):
        print("\nException : Sideview's \'Cloud Cover Layer\' option not found on the screen.")
        raise
    # Setting different levels and valid time
    if temp1 is not None and temp2 is not None:
        pag.click(temp1, temp2 + (gap * 4), interval=2)

    try:
        x, y = pag.locateCenterOnScreen(picture('sideviewwindow-valid.png'))
        pag.click(x + 200, y, interval=1)
        pag.move(0, 80, duration=1)
        pag.click(interval=1)
        pag.sleep(4)
    except (ImageNotFoundException, OSError, Exception):
        print("\nException : Sideview's \'Valid till\' button/option not found on the screen.")
        raise

    create_tutorial_images()
    # smaller region, seems the widget covers a bit the content
    pic = picture('sideviewwindow-cloud-cover-0-1-vertical-section-valid-'
                  '2012-10-18t06-00-00z-initialisation-2012-10-17t12-00-00z.png', boundingbox=(20, 20, 800, 200))
    loc = get_region(pic)
    sideview_region = (0, 0, loc.left + loc.width, loc.top)
    try:
        x, y = pag.locateCenterOnScreen(picture('sideviewwindow-mv-wp.png'), region=sideview_region)
        pag.click(x, y, interval=2)
        try:
            pic = picture('sideviewwindow-cloud-cover-0-1-vertical-section-valid-2012-10-18t06-00-00z-'
                          'initialisation-2012-10-17t12-00-00z.png', boundingbox=(103, 300, 118, 312))
            px, py = pag.locateCenterOnScreen(pic)
            # point1: 127, 394
        except (ImageNotFoundException, OSError, Exception):
            print("\nException : Sideview's \'Point 1\' not found on the screen.")
            raise
        offsets = [0, 114, 161, 200, ]
        for offset in offsets:
            pag.click(px + offset, py, interval=2)
            pag.moveTo(px + offset, py, duration=1)
            pag.dragTo(px + offset, py - offset - 50, duration=5, button='left')
            pag.click(interval=2)

    except ImageNotFoundException:
        print("\n Exception :Sideview's Move Waypoint button could not be located on the screen")
        raise

    create_tutorial_images()

    # Adding waypoints in SideView
    try:
        x, y = pag.locateCenterOnScreen(picture('sideviewwindow-ins-wp.png'), region=sideview_region)
        pag.click(x, y, duration=1)
        pag.click(x + 239, y + 186, duration=1)
        pag.sleep(3)
        pag.click(x + 383, y + 93, duration=1)
        pag.sleep(3)
        pag.click(x + 450, y + 140, duration=1)
        pag.sleep(4)
        pag.click(x, y, duration=1)
        pag.sleep(1)
    except (ImageNotFoundException, OSError, TypeError, Exception):
        print("\nException : Sideview's add waypoint button not found on the screen.")
        raise
    # Closing list layer of sideview and topview to make screen a little less congested.
    pag.click(ll_sv_x, ll_sv_y, duration=2)
    if platform == 'linux' or platform == 'linux2':
        pag.hotkey('altleft', 'f4')
    elif platform == 'win32':
        pag.hotkey('alt', 'f4')
    elif platform == 'darwin':
        pag.hotkey('command', 'w')
    pag.sleep(1)
    try:
        pag.click(ll_tov_x, ll_tov_y, duration=2)
        if platform == 'linux' or platform == 'linux2':
            pag.hotkey('altleft', 'f4')
        elif platform == 'win32':
            pag.hotkey('alt', 'f4')
        elif platform == 'darwin':
            pag.hotkey('command', 'w')
    except UnboundLocalError:
        # ToDo improve this
        pass

    # Table View
    # Opening Table View
    pag.move(-80, 120, duration=1)
    # pag.moveTo(1800, 1000, duration=1)
    pag.click(duration=1)
    # ANY now selected
    # ToDo ANY should be inactive whithout an OP
    pag.click(duration=1)

    pag.sleep(1)
    pag.hotkey('ctrl', 't')
    pag.sleep(2)

    create_tutorial_images()
    # Relocating Tableview and performing operations on table view
    # ToDo refactor to a module improve where it enters data
    try:
        x, y = pag.locateCenterOnScreen(picture('tableviewwindow-select-to-open-control.png'))
        pag.moveTo(x, y - 462, duration=1)
        if platform == 'linux' or platform == 'linux2':
            pag.dragRel(250, 887, duration=3)
        elif platform == 'win32' or platform == 'darwin':
            pag.dragRel(None, 487, duration=2)
        pag.sleep(2)
        if platform == 'linux' or platform == 'linux2':
            pag.keyDown('altleft')
            pag.press('tab')
            pag.press('tab')
            pag.keyUp('altleft')
            pag.sleep(1)
            pag.keyDown('altleft')
            pag.press('tab')
            pag.press('tab', presses=2)  # This needs to be checked in Linux
            pag.keyUp('altleft')
        elif platform == 'win32':
            pag.keyDown('alt')
            pag.press('tab')
            pag.press('right')
            pag.keyUp('alt')
            pag.sleep(1)
            pag.keyDown('alt')
            pag.press('tab')
            pag.press('right', presses=2)
            pag.keyUp('alt')
        elif platform == 'darwin':
            pag.keyDown('command')
            pag.press('tab')
            pag.press('right')
            pag.keyUp('command')
            pag.sleep(1)
            pag.keyDown('command')
            pag.press('tab')
            pag.press('right', presses=2)
            pag.keyUp('command')
        pag.sleep(1)
        if platform == 'win32' or platform == 'darwin':
            pag.dragRel(None, -300, duration=2)
            tv_x, tv_y = pag.position()
        elif platform == 'linux' or platform == 'linux2':
            pag.dragRel(None, -450, duration=2)
            tv_x, tv_y = pag.position()

        # Locating the selecttoopencontrol for tableview to perform operations
        try:
            x, y = pag.locateCenterOnScreen(picture('tableviewwindow-select-to-open-control.png'))
            xoffset = -50

            # Changing names of certain waypoints to predefined names
            pag.click(x + xoffset, y - 360, duration=1)
            pag.sleep(1)
            pag.doubleClick(duration=1)
            pag.sleep(2)
            pag.move(78, 0, duration=1)
            pag.sleep(1)
            pag.click(duration=1)
            pag.press('down', presses=5, interval=0.2)
            pag.sleep(1)
            pag.press('enter')
            pag.sleep(1)

            # Giving user defined names to waypoints
            pag.click(x + xoffset, y - 294, duration=1)
            pag.sleep(1)
            pag.doubleClick(duration=1)
            pag.sleep(1.5)
            if platform == 'linux' or platform == 'linux2' or platform == 'win32':
                pag.hotkey('ctrl', 'a')
            elif platform == 'darwin':
                pag.hotkey('command', 'a')
            pag.sleep(1)
            pag.write('Location A', interval=0.1)
            pag.sleep(1)
            pag.press('return') if platform == 'darwin' else pag.press('enter')
            pag.sleep(2)

            pag.click(x + xoffset, y - 263, duration=1)
            pag.sleep(1)
            pag.doubleClick(duration=1)
            pag.sleep(2)
            if platform == 'linux' or platform == 'linux2' or platform == 'win32':
                pag.hotkey('ctrl', 'a')
            elif platform == 'darwin':
                pag.hotkey('command', 'a')
            pag.sleep(1)
            pag.write('Stop Point', interval=0.1)
            pag.sleep(1)
            pag.press('return') if platform == 'darwin' else pag.press('enter')
            pag.sleep(2)

            # Changing Length of Flight Level
            pag.click(x + xoffset + 236, y - 263, duration=1)
            pag.sleep(1)
            pag.doubleClick(duration=1)
            pag.sleep(1)
            pag.write('319', interval=0.2)
            pag.sleep(1)
            pag.press('return') if platform == 'darwin' else pag.press('enter')
            pag.sleep(2)

            # Changing hPa level of waypoints
            pag.click(x + xoffset + 367, y - 232, duration=1)
            pag.sleep(1)
            pag.doubleClick(duration=1)
            pag.sleep(1)
            pag.write('250', interval=0.2)
            pag.sleep(1)
            pag.press('enter')
            pag.sleep(2)

            # Changing longitude of 'Location A' waypoint
            pag.click(x + xoffset + 165, y - 294, duration=1)
            pag.sleep(1)
            pag.doubleClick(duration=1)
            pag.sleep(1)
            pag.write('12.36', interval=0.2)
            pag.sleep(1)
            pag.press('return') if platform == 'darwin' else pag.press('enter')
            pag.sleep(2)

            # Cloning the row of waypoint
            try:
                x1, y1 = pag.locateCenterOnScreen(picture('tableviewwindow-clone.png'))
                pag.click(x + xoffset + 15, y - 263, duration=1)
                pag.sleep(1)
                pag.click(x1, y1, duration=1)
                pag.sleep(2)
                pag.click(x + xoffset + 15, y - 232, duration=1)
                pag.sleep(1)
                pag.click(x + xoffset + 117, y - 232, duration=1)
                pag.sleep(1)
                pag.write('65.26', interval=0.2)
                pag.sleep(1)
                pag.press('return') if platform == 'darwin' else pag.press('enter')
                pag.sleep(2)
                pag.move(580, 0, duration=1) if platform == 'win32' else pag.move(459, 0, duration=1)
                pag.doubleClick(duration=1)
                pag.sleep(2)
                pag.write('This is a reference comment', interval=0.2)
                pag.sleep(1)
                pag.press('return') if platform == 'darwin' else pag.press('enter')
                pag.sleep(2)
            except (ImageNotFoundException, OSError, TypeError, Exception):
                print("\nException : Tableview's CLONE button not found on the screen.")
                raise
            # Inserting a new row of waypoints
            try:
                x1, y1 = pag.locateCenterOnScreen(picture('tableviewwindow-insert.png'))
                pag.click(x + xoffset + 117, y - 294, duration=1)
                pag.sleep(2)
                pag.click(x1, y1, duration=1)
                pag.sleep(2)
                pag.click(x + xoffset + 117, y - 263, duration=1)
                pag.sleep(1)
                pag.doubleClick(duration=1)
                pag.sleep(1)
                pag.write('58', interval=0.2)
                pag.sleep(0.5)
                pag.press('return') if platform == 'darwin' else pag.press('enter')
                pag.sleep(2)
                pag.move(48, 0, duration=1)
                pag.doubleClick(duration=1)
                pag.sleep(1)
                pag.write('-1.64', interval=0.2)
                pag.sleep(1)
                pag.press('return') if platform == 'darwin' else pag.press('enter')
                pag.sleep(2)
                pag.move(108, 0, duration=1) if platform == 'win32' else pag.move(71, 0, duration=1)
                pag.doubleClick(duration=1)
                pag.sleep(1)
                pag.write('360', interval=0.2)
                pag.sleep(0.5)
                pag.press('return') if platform == 'darwin' else pag.press('enter')
                pag.sleep(2)
            except (ImageNotFoundException, OSError, TypeError, Exception):
                print("\nException : Tableview's INSERT button not found on the screen.")
                raise
            # Delete Selected waypoints row
            try:
                x1, y1 = pag.locateCenterOnScreen(picture('tableviewwindow-delete-selected.png'))
                pag.click(x + 150, y - 70, duration=1) if platform == 'win32' else pag.click(x + 150, y - 201,
                                                                                             duration=1)
                pag.sleep(2)
                pag.click(x1, y1, duration=1)
                pag.press('left')
                pag.sleep(1)
                pag.press('return') if platform == 'darwin' else pag.press('enter')
                pag.sleep(2)
            except (ImageNotFoundException, OSError, TypeError, Exception):
                print("\nException : Tableview's DELETE SELECTED button not found on the screen.")
                raise
            # Reverse waypoints' order
            try:
                x1, y1 = pag.locateCenterOnScreen(picture('tableviewwindow-reverse.png'))
                for _ in range(3):
                    pag.click(x1, y1, duration=1)
                    pag.sleep(1.5)
            except (ImageNotFoundException, OSError, TypeError, Exception):
                print("\nException : Tableview's REVERSE button not found on the screen.")
                raise
        except (ImageNotFoundException, OSError, TypeError, Exception):
            print("\nException : Tableview's selecttoopencontrol button (bottom part) not found on the screen.")
            raise
    except (ImageNotFoundException, OSError, TypeError, Exception):
        print("\nException : TableView's Select to open Control option (at the top) not found on the screen.")
        raise
    # Closing Table View to make space on screen
    if tv_x is not None and tv_y is not None:
        pag.click(tv_x, tv_y, duration=1)
    if platform == 'linux' or platform == 'linux2':
        pag.hotkey('altleft', 'f4')
        pag.press('left')
        pag.sleep(1)
        pag.press('enter')
    elif platform == 'win32':
        pag.hotkey('alt', 'f4')
        pag.press('left')
        pag.sleep(1)
        pag.press('enter')
    elif platform == 'darwin':
        pag.hotkey('command', 'w')
        pag.press('left')
        pag.sleep(1)
        pag.press('return')

    # Opening Linear View
    pag.sleep(1)
    pag.move(0, 400, duration=1)
    pag.click(interval=1)
    pag.hotkey('ctrl', 'l')
    pag.sleep(4)
    pag.hotkey(win, 'up')
    pag.click(10, 10, interval=2)
    pag.dragRel(853, 360, duration=3)
    pag.sleep(2)

    create_tutorial_images()
    # Relocating Linear View
    try:
        pag.locateCenterOnScreen(picture('linearwindow-select-to-open-control.png'))

        if platform == 'linux' or platform == 'linux2':
            pag.keyDown('altleft')
            pag.press('tab')
            pag.press('tab')
            pag.keyUp('altleft')
            pag.sleep(1)
            pag.keyDown('altleft')
            pag.press('tab')
            pag.press('tab')
            pag.press('tab')
            pag.keyUp('altleft')
        elif platform == 'win32':
            pag.keyDown('alt')
            pag.press('tab')
            pag.press('right')
            pag.keyUp('alt')
            pag.sleep(1)
            pag.keyDown('alt')
            pag.press('tab')
            pag.press('right', presses=2, interval=1)
            pag.keyUp('alt')
        elif platform == 'darwin':
            pag.keyDown('command')
            pag.press('tab')
            pag.press('right')
            pag.keyUp('command')
            pag.sleep(1)
            pag.keyDown('command')
            pag.press('tab')
            pag.press('right', presses=2, interval=1)
            pag.keyUp('command')
        pag.sleep(1)
        pag.dragRel(-102, -470, duration=2) if platform == 'win32' else pag.dragRel(-90, -500, duration=2)
        lv_x, lv_y = pag.position()
    except (ImageNotFoundException, OSError, TypeError, Exception):
        print("\nException : Linearview's window header not found on the screen.")
        raise

    # Locating Server Layer
    try:
        x, y = pag.locateCenterOnScreen(picture('linearwindow-server-layer.png'))
        pag.click(x, y, interval=2)

        # Entering wms URL
        try:
            x, y = pag.locateCenterOnScreen(picture('multilayersdialog-http-localhost-8081.png'))
            pag.click(x + 220, y, interval=2)
            pag.hotkey('ctrl', 'a', interval=1)
            pag.write('http://open-mss.org/', interval=0.25)
        except (ImageNotFoundException, OSError, Exception):
            print("\nException : Linearviews' \'WMS URL\' editbox button/option not found on the screen.")
            raise
        try:
            x, y = pag.locateCenterOnScreen(picture('multilayersdialog-get-capabilities.png'))
            pag.click(x, y, interval=2)
            pag.sleep(3)
        except (ImageNotFoundException, OSError, Exception):
            print("\nException : LinearView's \'Get capabilities\' button/option not found on the screen.")
            raise
        if platform == 'win32':
            pag.move(-171, -390, duration=1)
            pag.dragRel(-867, 135, duration=2)
        elif platform == 'linux' or platform == 'linux2' or platform == 'darwin':
            pag.move(-171, -390, duration=1)
            pag.dragRel(-900, 245, duration=2)
        # Storing screen coordinates for List layer of side view
        pag.position()
        pag.sleep(1)
    except (ImageNotFoundException, OSError, Exception):
        print("\nException : Linearview's WMS \'Server\\Layers\' button/option not found on the screen.")
        # raise
        pag.press('enter')

    create_tutorial_images()
    # Selecting Some Layers in Linear wms section
    if platform == 'win32':
        gap = 22
    elif platform == 'linux' or platform == 'linux2' or platform == 'darwin':
        gap = 16

    try:
        x, y = pag.locateCenterOnScreen(picture('multilayersdialog-multilayering.png'))
        # Cloudcover
        pag.click(x + 50, y + 70, interval=2)
        pag.sleep(1)
        temp1, temp2 = x, y
        pag.click(x, y, interval=2)
        pag.sleep(3)
        pag.move(0, gap, duration=1)
        pag.click(interval=1)
        pag.sleep(3)
        pag.move(0, gap * 2, duration=1)
        pag.click(interval=1)
        pag.sleep(3)
        pag.move(0, gap, duration=1)
        pag.click(interval=1)
        pag.sleep(3)
        pag.move(0, -gap * 4, duration=1)
        pag.click(interval=1)
        pag.sleep(3)
    except (ImageNotFoundException, OSError, Exception):
        print("\nException : Linearview's Lazer' option not found on the screen.")
        raise

    # Add waypoints after anaylzing the linear section wms
    try:
        x, y = pag.locateCenterOnScreen(picture('wms', 'add_waypoint.png'))
        pag.click(x, y, interval=2)
        pag.sleep(1)
        pag.click(x + 30, y + 50, duration=1)
        pag.sleep(2)
    except (ImageNotFoundException, OSError, Exception):
        print("\n Exception :Sideview's Add Waypoint button could not be located on the screen")
        raise

    # CLosing Linear View Layer List
    if temp1 is not None and temp2 is not None:
        pag.click(temp1, temp2 + (gap * 4), duration=2)
        pag.sleep(1)
        if platform == 'linux' or platform == 'linux2':
            pag.hotkey('altleft', 'f4')
        elif platform == 'win32':
            pag.hotkey('alt', 'f4')
        elif platform == 'darwin':
            pag.hotkey('command', 'w')
        pag.sleep(1)

    # Clicking on Linear View  Window Head
    if lv_x is not None and lv_y is not None:
        pag.click(lv_x, lv_y, duration=1)

    print("\nAutomation is over for this tutorial. Watch next tutorial for other functions.")

    # Close Everything!
    finish()


if __name__ == '__main__':
    start(target=automate_views, duration=567, dry_run=True)
