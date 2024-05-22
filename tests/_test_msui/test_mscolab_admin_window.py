# -*- coding: utf-8 -*-
"""

    tests._test_msui.test_mscolab_admin_window
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module is used to test mscolab-operation related gui.

    This file is part of MSS.

    :copyright: Copyright 2019 Shivashis Padhi
    :copyright: Copyright 2019-2024 by the MSS team, see AUTHORS.
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
import pytest

from mslib.mscolab.conf import mscolab_settings
from PyQt5 import QtCore, QtTest, QtWidgets
from mslib.msui import mscolab
from mslib.msui import msui
from mslib.mscolab.seed import add_user, get_user, add_operation, add_user_to_operation
from mslib.utils.config import modify_config_file


class Test_MscolabAdminWindow:
    @pytest.fixture(autouse=True)
    def setup(self, qtbot, mscolab_server):
        self.url = mscolab_server
        self.userdata = 'UV10@uv10', 'UV10', 'uv10'
        self.operation_name = "europe"
        assert add_user(self.userdata[0], self.userdata[1], self.userdata[2])
        assert add_operation(self.operation_name, "test europe")
        assert add_user_to_operation(path=self.operation_name, emailid=self.userdata[0], access_level="creator")
        self.user = get_user(self.userdata[0])
        assert add_user("collaborator@example.de", "example", "example")
        assert add_user_to_operation(path=self.operation_name,
                                     emailid="collaborator@example.de", access_level="collaborator")
        assert add_user("viewer@example.de", "viewer", "viewer")
        assert add_user_to_operation(path=self.operation_name, emailid="viewer@example.de", access_level="viewer")
        assert add_user("name1@example.de", "name1", "name1")
        assert add_user("name2@example.de", "name2", "name2")
        assert add_operation("paris", "test paris")
        assert add_user_to_operation(path="paris", emailid=self.userdata[0], access_level="creator")
        assert add_user_to_operation(path="paris", emailid="name1@example.de")
        assert add_operation("tokyo", "test tokyo")
        assert add_user_to_operation(path="tokyo", emailid=self.userdata[0], access_level="creator")

        self.window = msui.MSUIMainWindow(mscolab_data_dir=mscolab_settings.MSCOLAB_DATA_DIR)
        self.window.create_new_flight_track()
        self.window.show()
        # connect and login to mscolab
        self._connect_to_mscolab(qtbot)
        modify_config_file({"MSS_auth": {self.url: self.userdata[0]}})
        self._login(emailid=self.userdata[0], password=self.userdata[2])
        # activate operation and open chat window
        self._activate_operation_at_index(0)
        self.window.actionManageUsers.trigger()
        self.admin_window = self.window.mscolab.admin_window
        QtTest.QTest.qWaitForWindowExposed(self.window)
        yield
        self.window.mscolab.logout()
        if self.window.mscolab.admin_window:
            self.window.mscolab.admin_window.close()
        if self.window.mscolab.conn:
            self.window.mscolab.conn.disconnect()
        with mock.patch("PyQt5.QtWidgets.QMessageBox.warning", return_value=QtWidgets.QMessageBox.Yes):
            self.window.close()

    def test_permission_filter(self):
        len_added_users = self.admin_window.modifyUsersTable.rowCount()
        # Change filter to viewer
        self.admin_window.modifyUsersPermissionFilter.currentTextChanged.emit("viewer")
        # Check how many users are visible
        visible_row_count = self._get_visible_row_count(self.admin_window.modifyUsersTable)
        assert visible_row_count == 1
        # Change it back to all
        self.admin_window.modifyUsersPermissionFilter.currentTextChanged.emit("all")
        # Check how many rows are visible
        visible_row_count = self._get_visible_row_count(self.admin_window.modifyUsersTable)
        assert visible_row_count == len_added_users

    def test_text_search_filter(self):
        len_unadded_users = self.admin_window.addUsersTable.rowCount()
        len_added_users = self.admin_window.modifyUsersTable.rowCount()
        # Text Search in add users Table
        QtTest.QTest.keyClicks(self.admin_window.addUsersSearch, "name1")
        visible_row_count = self._get_visible_row_count(self.admin_window.addUsersTable)
        assert visible_row_count == 1
        self.admin_window.addUsersSearch.setText("")
        QtTest.QTest.keyClicks(self.admin_window.addUsersSearch, "")
        visible_row_count = self._get_visible_row_count(self.admin_window.addUsersTable)
        assert visible_row_count == len_unadded_users
        # Text Search in modify users Table
        QtTest.QTest.keyClicks(self.admin_window.modifyUsersSearch, "example")
        visible_row_count = self._get_visible_row_count(self.admin_window.modifyUsersTable)
        assert visible_row_count == 1
        self.admin_window.modifyUsersSearch.setText("")
        QtTest.QTest.keyClicks(self.admin_window.modifyUsersSearch, "")
        visible_row_count = self._get_visible_row_count(self.admin_window.modifyUsersTable)
        assert visible_row_count == len_added_users

    def test_permission_and_text_together(self):
        QtTest.QTest.keyClicks(self.admin_window.modifyUsersSearch, "viewer")
        self.admin_window.modifyUsersPermissionFilter.currentTextChanged.emit("viewer")
        visible_row_count = self._get_visible_row_count(self.admin_window.modifyUsersTable)
        assert visible_row_count == 1
        self.admin_window.modifyUsersPermissionFilter.currentTextChanged.emit("admin")
        visible_row_count = self._get_visible_row_count(self.admin_window.modifyUsersTable)
        assert visible_row_count == 0

    def test_add_permissions(self):
        len_unadded_users = self.admin_window.addUsersTable.rowCount()
        len_added_users = self.admin_window.modifyUsersTable.rowCount()
        users = ["name1", "name2"]
        # Select users in the add users table
        self._select_users(self.admin_window.addUsersTable, users)
        index = self.admin_window.addUsersPermission.findText("admin", QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.admin_window.addUsersPermission.setCurrentIndex(index)
        QtTest.QTest.mouseClick(self.admin_window.addUsersBtn, QtCore.Qt.LeftButton)
        # Check if they have been added in the modify users table
        self._check_users_present(self.admin_window.modifyUsersTable, users, "admin")
        assert len_unadded_users - 2 == self.admin_window.addUsersTable.rowCount()
        assert len_added_users + 2 == self.admin_window.modifyUsersTable.rowCount()

    def test_modify_permissions(self):
        users = ["name1", "name2"]
        # Select users in the add users table
        self._select_users(self.admin_window.addUsersTable, users)
        QtTest.QTest.mouseClick(self.admin_window.addUsersBtn, QtCore.Qt.LeftButton)
        # Select users in the modify users table
        self._select_users(self.admin_window.modifyUsersTable, users)
        # Update their permission to viewer
        index = self.admin_window.modifyUsersPermission.findText("viewer", QtCore.Qt.MatchFixedString)
        self.admin_window.modifyUsersPermission.setCurrentIndex(index)
        QtTest.QTest.mouseClick(self.admin_window.modifyUsersBtn, QtCore.Qt.LeftButton)
        # Check if the permission has been updated
        self._check_users_present(self.admin_window.modifyUsersTable, users, "viewer")

    def test_delete_permissions(self):
        # Select users in the add users table
        users = ["name1", "name2"]
        self._select_users(self.admin_window.addUsersTable, users)
        QtTest.QTest.mouseClick(self.admin_window.addUsersBtn, QtCore.Qt.LeftButton)
        len_unadded_users = self.admin_window.addUsersTable.rowCount()
        len_added_users = self.admin_window.modifyUsersTable.rowCount()

        # Select users in the modify users table
        self._select_users(self.admin_window.modifyUsersTable, users)
        # Click on delete permissions
        QtTest.QTest.mouseClick(self.admin_window.deleteUsersBtn, QtCore.Qt.LeftButton)
        # Check if the deleted users can be found in the add users table
        self._check_users_present(self.admin_window.addUsersTable, users)
        assert len_unadded_users + 2 == self.admin_window.addUsersTable.rowCount()
        assert len_added_users - 2 == self.admin_window.modifyUsersTable.rowCount()

    def test_import_permissions(self):
        index = self.admin_window.importPermissionsCB.findText("paris", QtCore.Qt.MatchFixedString)
        self.admin_window.importPermissionsCB.setCurrentIndex(index)
        QtTest.QTest.mouseClick(self.admin_window.importPermissionsBtn, QtCore.Qt.LeftButton)
        assert self.admin_window.modifyUsersTable.rowCount() == 1

    def _connect_to_mscolab(self, qtbot):
        self.connect_window = mscolab.MSColab_ConnectDialog(parent=self.window, mscolab=self.window.mscolab)
        self.window.mscolab.connect_window = self.connect_window
        self.connect_window.urlCb.setEditText(self.url)
        self.connect_window.show()
        QtTest.QTest.mouseClick(self.connect_window.connectBtn, QtCore.Qt.LeftButton)

        def assert_():
            assert not self.connect_window.connectBtn.isVisible()
            assert self.connect_window.disconnectBtn.isVisible()
        qtbot.wait_until(assert_)

    def _login(self, emailid, password):
        self.connect_window.loginEmailLe.setText(emailid)
        self.connect_window.loginPasswordLe.setText(password)
        QtTest.QTest.mouseClick(self.connect_window.loginBtn, QtCore.Qt.LeftButton)

    def _activate_operation_at_index(self, index):
        item = self.window.listOperationsMSC.item(index)
        point = self.window.listOperationsMSC.visualItemRect(item).center()
        QtTest.QTest.mouseClick(self.window.listOperationsMSC.viewport(), QtCore.Qt.LeftButton, pos=point)
        QtTest.QTest.mouseDClick(self.window.listOperationsMSC.viewport(), QtCore.Qt.LeftButton, pos=point)

    def _select_users(self, table, users):
        for row_num in range(table.rowCount()):
            item = table.item(row_num, 0)
            username = item.text()
            if username in users:
                point = table.visualItemRect(item).center()
                QtTest.QTest.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=point)
        assert len(table.selectionModel().selectedRows()) == 2

    def _get_visible_row_count(self, table):
        visible_row_count = 0
        for row_num in range(table.rowCount()):
            if table.isRowHidden(row_num) is False:
                visible_row_count += 1
        return visible_row_count

    def _check_users_present(self, table, users, access_level=None):
        found = 0
        for row_num in range(table.rowCount()):
            item = table.item(row_num, 0)
            username = item.text()
            if username in users:
                found += 1
                if access_level is not None:
                    assert table.item(row_num, 1).text() == access_level
        assert found == 2
