# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mslib/msui/ui/ui_mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MSSMainWindow(object):
    def setupUi(self, MSSMainWindow):
        MSSMainWindow.setObjectName("MSSMainWindow")
        MSSMainWindow.resize(738, 736)
        MSSMainWindow.setMinimumSize(QtCore.QSize(507, 736))
        self.centralwidget = QtWidgets.QWidget(MSSMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(8, 8, 8, 8)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName("gridLayout")
        self.openViewsGb = QtWidgets.QGroupBox(self.centralwidget)
        self.openViewsGb.setObjectName("openViewsGb")
        self.openViewsVL = QtWidgets.QVBoxLayout(self.openViewsGb)
        self.openViewsVL.setContentsMargins(8, 8, 8, 8)
        self.openViewsVL.setObjectName("openViewsVL")
        self.openViewsLabel = QtWidgets.QLabel(self.openViewsGb)
        self.openViewsLabel.setObjectName("openViewsLabel")
        self.openViewsVL.addWidget(self.openViewsLabel)
        self.listViews = QtWidgets.QListWidget(self.openViewsGb)
        self.listViews.setObjectName("listViews")
        self.openViewsVL.addWidget(self.listViews)
        self.gridLayout.addWidget(self.openViewsGb, 3, 0, 1, 1)
        self.openFlightTracksGb = QtWidgets.QGroupBox(self.centralwidget)
        self.openFlightTracksGb.setTitle("")
        self.openFlightTracksGb.setObjectName("openFlightTracksGb")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.openFlightTracksGb)
        self.verticalLayout.setContentsMargins(8, 8, 8, 8)
        self.verticalLayout.setObjectName("verticalLayout")
        self.openFlightTracksLabel = QtWidgets.QLabel(self.openFlightTracksGb)
        self.openFlightTracksLabel.setObjectName("openFlightTracksLabel")
        self.verticalLayout.addWidget(self.openFlightTracksLabel)
        self.listFlightTracks = QtWidgets.QListWidget(self.openFlightTracksGb)
        self.listFlightTracks.setObjectName("listFlightTracks")
        self.verticalLayout.addWidget(self.listFlightTracks)
        self.gridLayout.addWidget(self.openFlightTracksGb, 2, 0, 1, 1)
        self.userOptionsHL = QtWidgets.QHBoxLayout()
        self.userOptionsHL.setContentsMargins(0, -1, 0, -1)
        self.userOptionsHL.setObjectName("userOptionsHL")
        self.mscStatusLabel = QtWidgets.QLabel(self.centralwidget)
        self.mscStatusLabel.setWordWrap(True)
        self.mscStatusLabel.setObjectName("mscStatusLabel")
        self.userOptionsHL.addWidget(self.mscStatusLabel)
        self.usernameLabel = QtWidgets.QLabel(self.centralwidget)
        self.usernameLabel.setObjectName("usernameLabel")
        self.userOptionsHL.addWidget(self.usernameLabel, 0, QtCore.Qt.AlignRight)
        self.userOptionsTb = QtWidgets.QToolButton(self.centralwidget)
        self.userOptionsTb.setStyleSheet("::menu-indicator { image: none; }")
        self.userOptionsTb.setText("")
        self.userOptionsTb.setObjectName("userOptionsTb")
        self.userOptionsHL.addWidget(self.userOptionsTb, 0, QtCore.Qt.AlignRight)
        self.connectBtn = QtWidgets.QPushButton(self.centralwidget)
        self.connectBtn.setObjectName("connectBtn")
        self.userOptionsHL.addWidget(self.connectBtn, 0, QtCore.Qt.AlignRight)
        self.userOptionsHL.setStretch(0, 1)
        self.gridLayout.addLayout(self.userOptionsHL, 0, 0, 1, 2)
        self.openOperationsGb = QtWidgets.QGroupBox(self.centralwidget)
        self.openOperationsGb.setTitle("")
        self.openOperationsGb.setObjectName("openOperationsGb")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.openOperationsGb)
        self.gridLayout_3.setContentsMargins(8, 8, 8, 8)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.workingStatusLabel = QtWidgets.QLabel(self.openOperationsGb)
        self.workingStatusLabel.setWordWrap(True)
        self.workingStatusLabel.setObjectName("workingStatusLabel")
        self.gridLayout_3.addWidget(self.workingStatusLabel, 2, 0, 1, 2)
        self.listOperationsMSC = QtWidgets.QListWidget(self.openOperationsGb)
        self.listOperationsMSC.setObjectName("listOperationsMSC")
        self.gridLayout_3.addWidget(self.listOperationsMSC, 1, 0, 1, 2)
        self.serverOptionsCb = QtWidgets.QComboBox(self.openOperationsGb)
        self.serverOptionsCb.setObjectName("serverOptionsCb")
        self.serverOptionsCb.addItem("")
        self.serverOptionsCb.addItem("")
        self.serverOptionsCb.addItem("")
        self.gridLayout_3.addWidget(self.serverOptionsCb, 6, 1, 1, 1)
        self.openOperationsMSCLabel = QtWidgets.QLabel(self.openOperationsGb)
        self.openOperationsMSCLabel.setObjectName("openOperationsMSCLabel")
        self.gridLayout_3.addWidget(self.openOperationsMSCLabel, 0, 0, 1, 1)
        self.categoryLabel = QtWidgets.QLabel(self.openOperationsGb)
        self.categoryLabel.setObjectName("categoryLabel")
        self.gridLayout_3.addWidget(self.categoryLabel, 5, 0, 1, 1)
        self.workLocallyCheckbox = QtWidgets.QCheckBox(self.openOperationsGb)
        self.workLocallyCheckbox.setObjectName("workLocallyCheckbox")
        self.gridLayout_3.addWidget(self.workLocallyCheckbox, 6, 0, 1, 1)
        self.filterCategoryCb = QtWidgets.QComboBox(self.openOperationsGb)
        self.filterCategoryCb.setAutoFillBackground(False)
        self.filterCategoryCb.setEditable(False)
        self.filterCategoryCb.setObjectName("filterCategoryCb")
        self.filterCategoryCb.addItem("")
        self.gridLayout_3.addWidget(self.filterCategoryCb, 5, 1, 1, 1)
        self.gridLayout.addWidget(self.openOperationsGb, 2, 1, 2, 1)
        self.activeOperationDesc = QtWidgets.QLabel(self.centralwidget)
        self.activeOperationDesc.setObjectName("activeOperationDesc")
        self.gridLayout.addWidget(self.activeOperationDesc, 1, 0, 1, 2)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        MSSMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MSSMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 738, 26))
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuImportFlightTrack = QtWidgets.QMenu(self.menuFile)
        self.menuImportFlightTrack.setObjectName("menuImportFlightTrack")
        self.menuExportActiveFlightTrack = QtWidgets.QMenu(self.menuFile)
        self.menuExportActiveFlightTrack.setObjectName("menuExportActiveFlightTrack")
        self.menuNew = QtWidgets.QMenu(self.menuFile)
        self.menuNew.setObjectName("menuNew")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuViews = QtWidgets.QMenu(self.menubar)
        self.menuViews.setObjectName("menuViews")
        self.menuOperation = QtWidgets.QMenu(self.menubar)
        self.menuOperation.setObjectName("menuOperation")
        self.menuProperties = QtWidgets.QMenu(self.menuOperation)
        self.menuProperties.setObjectName("menuProperties")
        MSSMainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MSSMainWindow)
        self.statusBar.setObjectName("statusBar")
        MSSMainWindow.setStatusBar(self.statusBar)
        self.actionSaveActiveFlightTrack = QtWidgets.QAction(MSSMainWindow)
        self.actionSaveActiveFlightTrack.setObjectName("actionSaveActiveFlightTrack")
        self.actionSaveActiveFlightTrackAs = QtWidgets.QAction(MSSMainWindow)
        self.actionSaveActiveFlightTrackAs.setObjectName("actionSaveActiveFlightTrackAs")
        self.actionAboutMSUI = QtWidgets.QAction(MSSMainWindow)
        self.actionAboutMSUI.setObjectName("actionAboutMSUI")
        self.actionOnlineHelp = QtWidgets.QAction(MSSMainWindow)
        self.actionOnlineHelp.setObjectName("actionOnlineHelp")
        self.actionQuit = QtWidgets.QAction(MSSMainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionShortcuts = QtWidgets.QAction(MSSMainWindow)
        self.actionShortcuts.setObjectName("actionShortcuts")
        self.actionCloseSelectedFlightTrack = QtWidgets.QAction(MSSMainWindow)
        self.actionCloseSelectedFlightTrack.setObjectName("actionCloseSelectedFlightTrack")
        self.actionUpdater = QtWidgets.QAction(MSSMainWindow)
        self.actionUpdater.setObjectName("actionUpdater")
        self.actionConfiguration = QtWidgets.QAction(MSSMainWindow)
        self.actionConfiguration.setObjectName("actionConfiguration")
        self.actionTopView = QtWidgets.QAction(MSSMainWindow)
        self.actionTopView.setObjectName("actionTopView")
        self.actionSideView = QtWidgets.QAction(MSSMainWindow)
        self.actionSideView.setObjectName("actionSideView")
        self.actionTableView = QtWidgets.QAction(MSSMainWindow)
        self.actionTableView.setObjectName("actionTableView")
        self.actionLinearView = QtWidgets.QAction(MSSMainWindow)
        self.actionLinearView.setObjectName("actionLinearView")
        self.actionChat = QtWidgets.QAction(MSSMainWindow)
        self.actionChat.setObjectName("actionChat")
        self.actionVersionHistory = QtWidgets.QAction(MSSMainWindow)
        self.actionVersionHistory.setObjectName("actionVersionHistory")
        self.actionManageUsers = QtWidgets.QAction(MSSMainWindow)
        self.actionManageUsers.setObjectName("actionManageUsers")
        self.actionMSColabHelp = QtWidgets.QAction(MSSMainWindow)
        self.actionMSColabHelp.setObjectName("actionMSColabHelp")
        self.actionDeleteOperation = QtWidgets.QAction(MSSMainWindow)
        self.actionDeleteOperation.setObjectName("actionDeleteOperation")
        self.actionBringMainWindowToFront = QtWidgets.QAction(MSSMainWindow)
        self.actionBringMainWindowToFront.setObjectName("actionBringMainWindowToFront")
        self.actionNewFlightTrack = QtWidgets.QAction(MSSMainWindow)
        self.actionNewFlightTrack.setObjectName("actionNewFlightTrack")
        self.actionAddOperation = QtWidgets.QAction(MSSMainWindow)
        self.actionAddOperation.setObjectName("actionAddOperation")
        self.actionSearch = QtWidgets.QAction(MSSMainWindow)
        self.actionSearch.setObjectName("actionSearch")
        self.menuNew.addAction(self.actionNewFlightTrack)
        self.menuNew.addAction(self.actionAddOperation)
        self.menuFile.addAction(self.menuNew.menuAction())
        self.menuFile.addAction(self.menuImportFlightTrack.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSaveActiveFlightTrack)
        self.menuFile.addAction(self.actionSaveActiveFlightTrackAs)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionCloseSelectedFlightTrack)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.menuExportActiveFlightTrack.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionConfiguration)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menuHelp.addAction(self.actionShortcuts)
        self.menuHelp.addAction(self.actionMSColabHelp)
        self.menuHelp.addAction(self.actionUpdater)
        self.menuHelp.addAction(self.actionOnlineHelp)
        self.menuHelp.addAction(self.actionAboutMSUI)
        self.menuHelp.addAction(self.actionSearch)
        self.menuViews.addAction(self.actionTopView)
        self.menuViews.addAction(self.actionSideView)
        self.menuViews.addAction(self.actionTableView)
        self.menuViews.addAction(self.actionLinearView)
        self.menuProperties.addAction(self.actionDeleteOperation)
        self.menuOperation.addAction(self.actionChat)
        self.menuOperation.addAction(self.actionVersionHistory)
        self.menuOperation.addAction(self.actionManageUsers)
        self.menuOperation.addSeparator()
        self.menuOperation.addAction(self.menuProperties.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuViews.menuAction())
        self.menubar.addAction(self.menuOperation.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MSSMainWindow)
        self.filterCategoryCb.setCurrentIndex(0)
        self.actionQuit.triggered.connect(MSSMainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MSSMainWindow)
        MSSMainWindow.setTabOrder(self.connectBtn, self.userOptionsTb)
        MSSMainWindow.setTabOrder(self.userOptionsTb, self.listFlightTracks)
        MSSMainWindow.setTabOrder(self.listFlightTracks, self.listViews)
        MSSMainWindow.setTabOrder(self.listViews, self.listOperationsMSC)
        MSSMainWindow.setTabOrder(self.listOperationsMSC, self.workLocallyCheckbox)
        MSSMainWindow.setTabOrder(self.workLocallyCheckbox, self.serverOptionsCb)

    def retranslateUi(self, MSSMainWindow):
        _translate = QtCore.QCoreApplication.translate
        MSSMainWindow.setWindowTitle(_translate("MSSMainWindow", "Mission Support System"))
        self.openViewsLabel.setText(_translate("MSSMainWindow", "Open Views:"))
        self.listViews.setToolTip(_translate("MSSMainWindow", "Double-click a view to bring it to the front."))
        self.openFlightTracksLabel.setText(_translate("MSSMainWindow", "Flight Tracks:"))
        self.listFlightTracks.setToolTip(_translate("MSSMainWindow", "List of open flight tracks.\n"
"Double-click a flight track to activate it.\n"
"Save a flight track to name it."))
        self.mscStatusLabel.setText(_translate("MSSMainWindow", "Status: Disconnected"))
        self.usernameLabel.setText(_translate("MSSMainWindow", "User"))
        self.userOptionsTb.setToolTip(_translate("MSSMainWindow", "Profile options"))
        self.connectBtn.setToolTip(_translate("MSSMainWindow", "Connect to an MSColab Server"))
        self.connectBtn.setText(_translate("MSSMainWindow", "Connect to MSColab"))
        self.workingStatusLabel.setText(_translate("MSSMainWindow", "No operations selected"))
        self.listOperationsMSC.setToolTip(_translate("MSSMainWindow", "List of mscolab operations."))
        self.serverOptionsCb.setToolTip(_translate("MSSMainWindow", "Fetch/Save Server options"))
        self.serverOptionsCb.setItemText(0, _translate("MSSMainWindow", "Server Options"))
        self.serverOptionsCb.setItemText(1, _translate("MSSMainWindow", "Fetch From Server"))
        self.serverOptionsCb.setItemText(2, _translate("MSSMainWindow", "Save To Server"))
        self.openOperationsMSCLabel.setText(_translate("MSSMainWindow", "Operations:"))
        self.categoryLabel.setText(_translate("MSSMainWindow", "Category:"))
        self.workLocallyCheckbox.setToolTip(_translate("MSSMainWindow", "Check to work asynchronously from the server"))
        self.workLocallyCheckbox.setText(_translate("MSSMainWindow", "Work Asynchronously"))
        self.filterCategoryCb.setWhatsThis(_translate("MSSMainWindow", "filter by operation category"))
        self.filterCategoryCb.setCurrentText(_translate("MSSMainWindow", "ANY"))
        self.filterCategoryCb.setItemText(0, _translate("MSSMainWindow", "ANY"))
        self.activeOperationDesc.setText(_translate("MSSMainWindow", "Select Operation to View Description."))
        self.menuFile.setTitle(_translate("MSSMainWindow", "&File"))
        self.menuImportFlightTrack.setTitle(_translate("MSSMainWindow", "Import Flight Track"))
        self.menuExportActiveFlightTrack.setTitle(_translate("MSSMainWindow", "Export Flight Track"))
        self.menuNew.setTitle(_translate("MSSMainWindow", "New"))
        self.menuHelp.setTitle(_translate("MSSMainWindow", "&Help"))
        self.menuViews.setTitle(_translate("MSSMainWindow", "Views"))
        self.menuOperation.setTitle(_translate("MSSMainWindow", "Operation"))
        self.menuProperties.setTitle(_translate("MSSMainWindow", "Properties"))
        self.actionSaveActiveFlightTrack.setText(_translate("MSSMainWindow", "&Save Active Flight Track"))
        self.actionSaveActiveFlightTrack.setShortcut(_translate("MSSMainWindow", "Ctrl+S"))
        self.actionSaveActiveFlightTrackAs.setText(_translate("MSSMainWindow", "Save Active Flight Track As"))
        self.actionSaveActiveFlightTrackAs.setShortcut(_translate("MSSMainWindow", "Ctrl+Shift+S"))
        self.actionAboutMSUI.setText(_translate("MSSMainWindow", "&About MSUI"))
        self.actionOnlineHelp.setText(_translate("MSSMainWindow", "&Online Help"))
        self.actionQuit.setText(_translate("MSSMainWindow", "&Quit"))
        self.actionQuit.setShortcut(_translate("MSSMainWindow", "Ctrl+Q"))
        self.actionShortcuts.setText(_translate("MSSMainWindow", "&Shortcuts"))
        self.actionShortcuts.setToolTip(_translate("MSSMainWindow", "Open the shortcuts dialog"))
        self.actionShortcuts.setShortcut(_translate("MSSMainWindow", "Alt+S"))
        self.actionCloseSelectedFlightTrack.setText(_translate("MSSMainWindow", "&Close Selected Local Flight Track"))
        self.actionUpdater.setText(_translate("MSSMainWindow", "&Updater"))
        self.actionConfiguration.setText(_translate("MSSMainWindow", "&Configuration"))
        self.actionTopView.setText(_translate("MSSMainWindow", "&Top View (Horizontal Section)"))
        self.actionTopView.setShortcut(_translate("MSSMainWindow", "Ctrl+H"))
        self.actionSideView.setText(_translate("MSSMainWindow", "&Side View (Vertical Section)"))
        self.actionSideView.setShortcut(_translate("MSSMainWindow", "Ctrl+V"))
        self.actionTableView.setText(_translate("MSSMainWindow", "&Table View"))
        self.actionTableView.setShortcut(_translate("MSSMainWindow", "Ctrl+T"))
        self.actionLinearView.setText(_translate("MSSMainWindow", "&Linear View"))
        self.actionLinearView.setShortcut(_translate("MSSMainWindow", "Ctrl+L"))
        self.actionChat.setText(_translate("MSSMainWindow", "&Chat"))
        self.actionVersionHistory.setText(_translate("MSSMainWindow", "&Version History"))
        self.actionManageUsers.setText(_translate("MSSMainWindow", "&Manage Users"))
        self.actionMSColabHelp.setText(_translate("MSSMainWindow", "&MSColab"))
        self.actionDeleteOperation.setText(_translate("MSSMainWindow", "&Delete Operation"))
        self.actionBringMainWindowToFront.setText(_translate("MSSMainWindow", "Bring Main Window To Front"))
        self.actionBringMainWindowToFront.setShortcut(_translate("MSSMainWindow", "Ctrl+Up"))
        self.actionNewFlightTrack.setText(_translate("MSSMainWindow", "&Local Flight Track"))
        self.actionAddOperation.setText(_translate("MSSMainWindow", "&MSColab Operation"))
        self.actionSearch.setText(_translate("MSSMainWindow", "Search"))
        self.actionSearch.setToolTip(_translate("MSSMainWindow", "Search for interactive text in the UI"))
        self.actionSearch.setShortcut(_translate("MSSMainWindow", "Ctrl+F"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MSSMainWindow = QtWidgets.QMainWindow()
    ui = Ui_MSSMainWindow()
    ui.setupUi(MSSMainWindow)
    MSSMainWindow.show()
    sys.exit(app.exec_())
