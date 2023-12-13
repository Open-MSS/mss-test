# -*- coding: utf-8 -*-
"""

    tests._test_msui.test_topview
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module provides pytest functions to tests msui.topview

    This file is part of MSS.

    :copyright: Copyright 2017 Joern Ungermann
    :copyright: Copyright 2017-2023 by the MSS team, see AUTHORS.
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

import mock
import os
import pytest
import shutil
import tempfile
import mslib.msui.topview as tv
from PyQt5 import QtWidgets, QtCore, QtTest
from mslib.msui import flighttrack as ft
from mslib.msui.msui import MSUIMainWindow
from mslib.msui.mpl_qtwidget import _DEFAULT_SETTINGS_TOPVIEW
from tests.utils import qt_wait_until


class Test_MSS_TV_MapAppearanceDialog(object):
    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        self.window = tv.MSUI_TV_MapAppearanceDialog(settings=_DEFAULT_SETTINGS_TOPVIEW)
        self.window.show()
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.qWaitForWindowExposed(self.window)
        QtWidgets.QApplication.processEvents()
        yield
        self.window.close()
        self.window.deleteLater()
        QtWidgets.QApplication.processEvents()

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_show(self, mockcrit):
        assert mockcrit.critical.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_get(self, mockcrit):
        assert mockcrit.critical.call_count == 0
        self.window.get_settings()
        assert mockcrit.critical.call_count == 0


class Test_MSSTopViewWindow(object):
    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        self.main_window = MSUIMainWindow()
        initial_waypoints = [ft.Waypoint(40., 25., 0), ft.Waypoint(60., -10., 0), ft.Waypoint(40., 10, 0)]
        waypoints_model = ft.WaypointsTableModel("")
        waypoints_model.insertRows(
            0, rows=len(initial_waypoints), waypoints=initial_waypoints)
        self.window = tv.MSUITopViewWindow(model=waypoints_model, mainwindow=self.main_window)
        self.window.show()
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.qWaitForWindowExposed(self.window)
        QtWidgets.QApplication.processEvents()
        yield
        with mock.patch("PyQt5.QtWidgets.QMessageBox.warning", return_value=QtWidgets.QMessageBox.Yes):
            self.main_window.close()
        self.main_window.deleteLater()
        QtWidgets.QApplication.processEvents()

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_open_wms(self, mockbox):
        self.window.cbTools.currentIndexChanged.emit(1)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_open_sat(self, mockbox):
        self.window.cbTools.currentIndexChanged.emit(2)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_open_rs(self, mockcrit):
        self.window.cbTools.currentIndexChanged.emit(3)
        QtWidgets.QApplication.processEvents()
        rsdock = self.window.docks[2].widget()
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mouseClick(rsdock.cbDrawTangents, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        rsdock.dsbTangentHeight.setValue(6)
        QtWidgets.QApplication.processEvents()
        rsdock.dsbObsAngleAzimuth.setValue(70)
        QtTest.QTest.mouseClick(rsdock.cbDrawTangents, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        rsdock.cbShowSolarAngle.setChecked(True)
        assert mockcrit.critical.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_open_kml(self, mockbox):
        self.window.cbTools.currentIndexChanged.emit(4)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_insert_point(self, mockbox):
        """
        Test inserting a point inside and outside the canvas
        """
        self.window.mpl.navbar._actions['insert_wp'].trigger()
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 3
        QtTest.QTest.mouseClick(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 4
        QtTest.QTest.mouseClick(self.window.mpl.canvas, QtCore.Qt.LeftButton, pos=QtCore.QPoint(1, 1))
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 4
        QtTest.QTest.mouseClick(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        # click again on same position
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 5
        assert mockbox.critical.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox.question",
                return_value=QtWidgets.QMessageBox.Yes)
    @mock.patch("PyQt5.QtWidgets.QMessageBox.critical")
    def test_remove_point_yes(self, mockcrit, mockbox):
        self.window.mpl.navbar._actions['insert_wp'].trigger()
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 3
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mouseClick(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 4
        self.window.mpl.navbar._actions['delete_wp'].trigger()
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mouseClick(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        assert mockcrit.call_count == 0
        assert len(self.window.waypoints_model.waypoints) == 3
        assert mockbox.call_count == 1

    @mock.patch("PyQt5.QtWidgets.QMessageBox.question",
                return_value=QtWidgets.QMessageBox.No)
    @mock.patch("PyQt5.QtWidgets.QMessageBox.critical")
    def test_remove_point_no(self, mockcrit, mockbox):
        self.window.mpl.navbar._actions['insert_wp'].trigger()
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 3
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mouseClick(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 4
        self.window.mpl.navbar._actions['delete_wp'].trigger()
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mousePress(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mouseRelease(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        assert mockbox.call_count == 1
        assert len(self.window.waypoints_model.waypoints) == 4
        assert mockcrit.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_move_point(self, mockbox):
        self.window.mpl.navbar._actions['insert_wp'].trigger()
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 3
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mouseClick(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 4
        self.window.mpl.navbar._actions['move_wp'].trigger()
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mousePress(self.window.mpl.canvas, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        point = QtCore.QPoint((self.window.width() // 3), self.window.height() // 2)
        QtTest.QTest.mouseMove(
            self.window.mpl.canvas, pos=point)
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mouseRelease(
            self.window.mpl.canvas, QtCore.Qt.LeftButton, pos=point)
        QtWidgets.QApplication.processEvents()
        assert len(self.window.waypoints_model.waypoints) == 4
        assert mockbox.critical.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_roundtrip(self, mockbox):
        """
        Test connecting the last and first point
        Test connecting the first point to itself
        """
        count = len(self.window.waypoints_model.waypoints)

        # Test if the last waypoint connects to the first
        self.window.update_roundtrip_enabled()
        assert self.window.is_roundtrip_possible()
        self.window.make_roundtrip()
        assert len(self.window.waypoints_model.waypoints) == count + 1
        first = self.window.waypoints_model.waypoints[0]
        dupe = self.window.waypoints_model.waypoints[-1]
        assert first.lat == dupe.lat and first.lon == dupe.lon

        # Check if roundtrip is disabled if the last and first point are equal
        self.window.update_roundtrip_enabled()
        assert not self.window.is_roundtrip_possible()
        assert not self.window.btRoundtrip.isEnabled()
        self.window.make_roundtrip()
        assert len(self.window.waypoints_model.waypoints) == count + 1

        # Remove connection
        self.window.waypoints_model.removeRows(count, 1)
        assert len(self.window.waypoints_model.waypoints) == count
        assert mockbox.critical.call_count == 0

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_map_options(self, mockbox):
        self.window.mpl.canvas.map.set_graticule_visible(True)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0
        self.window.mpl.canvas.map.set_graticule_visible(False)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0
        self.window.mpl.canvas.map.set_fillcontinents_visible(False)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0
        self.window.mpl.canvas.map.set_fillcontinents_visible(True)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0
        self.window.mpl.canvas.map.set_coastlines_visible(False)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0
        self.window.mpl.canvas.map.set_coastlines_visible(True)
        QtWidgets.QApplication.processEvents()
        assert mockbox.critical.call_count == 0

        with mock.patch("mslib.msui.mpl_map.get_airports", return_value=[{"type": "small_airport", "name": "Test",
                                                                          "latitude_deg": 52, "longitude_deg": 13,
                                                                          "elevation_ft": 0}]):
            self.window.mpl.canvas.map.set_draw_airports(True)
            QtWidgets.QApplication.processEvents()
            assert mockbox.critical.call_count == 0
        with mock.patch("mslib.msui.mpl_map.get_airports", return_value=[]):
            self.window.mpl.canvas.map.set_draw_airports(True)
            QtWidgets.QApplication.processEvents()
            assert mockbox.critical.call_count == 0
        with mock.patch("mslib.msui.mpl_map.get_airports", return_value=[{"type": "small_airport", "name": "Test",
                                                                          "latitude_deg": -52, "longitude_deg": -13,
                                                                          "elevation_ft": 0}]):
            self.window.mpl.canvas.map.set_draw_airports(True)
            QtWidgets.QApplication.processEvents()
            assert mockbox.critical.call_count == 0

        with mock.patch("mslib.msui.mpl_map.get_airspaces", return_value=[{"name": "Test", "top": 1, "bottom": 0,
                                                                           "polygon": [(13, 52), (14, 53), (13, 52)],
                                                                           "country": "DE"}]):
            self.window.mpl.canvas.map.set_draw_airspaces(True)
            QtWidgets.QApplication.processEvents()
            assert mockbox.critical.call_count == 0
        with mock.patch("mslib.msui.mpl_map.get_airspaces", return_value=[]):
            self.window.mpl.canvas.map.set_draw_airspaces(True)
            QtWidgets.QApplication.processEvents()
            assert mockbox.critical.call_count == 0
        with mock.patch("mslib.msui.mpl_map.get_airspaces", return_value=[{"name": "Test", "top": 1, "bottom": 0,
                                                                           "polygon": [(-13, -52), (-14, -53),
                                                                                       (-13, -52)],
                                                                           "country": "DE"}]):
            self.window.mpl.canvas.map.set_draw_airspaces(True)
            QtWidgets.QApplication.processEvents()
            assert mockbox.critical.call_count == 0


class Test_TopViewWMS(object):
    @pytest.fixture(autouse=True)
    def setup(self, mswms_server, qapp):
        self.url = mswms_server

        self.tempdir = tempfile.mkdtemp()
        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)

        initial_waypoints = [ft.Waypoint(40., 25., 0), ft.Waypoint(60., -10., 0), ft.Waypoint(40., 10, 0)]
        waypoints_model = ft.WaypointsTableModel("")
        waypoints_model.insertRows(
            0, rows=len(initial_waypoints), waypoints=initial_waypoints)

        self.main_window = MSUIMainWindow()
        self.window = tv.MSUITopViewWindow(model=waypoints_model, mainwindow=self.main_window)
        self.window.show()
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.qWait(2000)
        QtTest.QTest.qWaitForWindowExposed(self.window)
        QtWidgets.QApplication.processEvents()
        self.window.cbTools.currentIndexChanged.emit(1)
        QtWidgets.QApplication.processEvents()
        self.wms_control = self.window.docks[0].widget()
        self.wms_control.multilayers.cbWMS_URL.setEditText("")
        yield
        with mock.patch("PyQt5.QtWidgets.QMessageBox.warning", return_value=QtWidgets.QMessageBox.Yes):
            self.main_window.close()
        self.main_window.deleteLater()
        QtWidgets.QApplication.processEvents()
        shutil.rmtree(self.tempdir)

    def query_server(self, url):
        cpdlg_canceled_spy = QtTest.QSignalSpy(self.wms_control.cpdlg.canceled)
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.keyClicks(self.wms_control.multilayers.cbWMS_URL, url)
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.mouseClick(self.wms_control.multilayers.btGetCapabilities, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        cpdlg_canceled_spy.wait()

    @mock.patch("PyQt5.QtWidgets.QMessageBox")
    def test_server_getmap(self, mockbox):
        """
        assert that a getmap call to a WMS server displays an image
        """
        self.query_server(self.url)
        QtTest.QTest.mouseClick(self.wms_control.btGetMap, QtCore.Qt.LeftButton)
        QtWidgets.QApplication.processEvents()
        qt_wait_until(lambda: self.window.getView().map.image is not None)
        assert self.window.getView().map.image is not None
        self.window.getView().set_settings({})
        self.window.getView().clear_figure()
        qt_wait_until(lambda: self.window.getView().map.image is None)
        assert self.window.getView().map.image is None
        self.window.mpl.canvas.redraw_map()
        assert mockbox.critical.call_count == 0


class Test_MSUITopViewWindow():
    @pytest.fixture(autouse=True)
    def setup(self, qapp):
        pass

    def test_kwargs_update_does_not_harm(self):
        initial_waypoints = [ft.Waypoint(40., 25., 0), ft.Waypoint(60., -10., 0), ft.Waypoint(40., 10, 0)]
        waypoints_model = ft.WaypointsTableModel("")
        waypoints_model.insertRows(0, rows=len(initial_waypoints), waypoints=initial_waypoints)
        mainwindow = MSUIMainWindow()
        self.window = tv.MSUITopViewWindow(model=waypoints_model, mainwindow=mainwindow)

        # user_options is a global var
        from mslib.utils.config import user_options

        assert user_options['predefined_map_sections']['01 Europe (cyl)']['map'] == {'llcrnrlat': 35.0,
                                                                                     'llcrnrlon': -15.0,
                                                                                     'urcrnrlat': 65.0,
                                                                                     'urcrnrlon': 30.0}

        with mock.patch("PyQt5.QtWidgets.QMessageBox.warning", return_value=QtWidgets.QMessageBox.Yes):
            mainwindow.close()
        mainwindow.deleteLater()
