# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mslib/msui/ui/ui_mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MSUIMainWindow(object):
    def setupUi(self, MSUIMainWindow):
        MSUIMainWindow.setObjectName("MSUIMainWindow")
        MSUIMainWindow.resize(738, 749)
        MSUIMainWindow.setMinimumSize(QtCore.QSize(507, 736))
        self.centralwidget = QtWidgets.QWidget(MSUIMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(8, 8, 8, 8)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.MSColabConnectGb = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MSColabConnectGb.sizePolicy().hasHeightForWidth())
        self.MSColabConnectGb.setSizePolicy(sizePolicy)
        self.MSColabConnectGb.setMinimumSize(QtCore.QSize(0, 0))
        self.MSColabConnectGb.setTitle("")
        self.MSColabConnectGb.setObjectName("MSColabConnectGb")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.MSColabConnectGb)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.mscolabServerLabel = QtWidgets.QLabel(self.MSColabConnectGb)
        self.mscolabServerLabel.setObjectName("mscolabServerLabel")
        self.verticalLayout_3.addWidget(self.mscolabServerLabel)
        self.userOptionsHL = QtWidgets.QHBoxLayout()
        self.userOptionsHL.setContentsMargins(0, -1, 0, -1)
        self.userOptionsHL.setObjectName("userOptionsHL")
        self.mscStatusLabel = QtWidgets.QLabel(self.MSColabConnectGb)
        self.mscStatusLabel.setWordWrap(True)
        self.mscStatusLabel.setObjectName("mscStatusLabel")
        self.userOptionsHL.addWidget(self.mscStatusLabel)
        self.usernameLabel = QtWidgets.QLabel(self.MSColabConnectGb)
        self.usernameLabel.setObjectName("usernameLabel")
        self.userOptionsHL.addWidget(self.usernameLabel, 0, QtCore.Qt.AlignRight)
        self.userOptionsTb = QtWidgets.QToolButton(self.MSColabConnectGb)
        self.userOptionsTb.setStyleSheet("::menu-indicator { image: none; }")
        self.userOptionsTb.setText("")
        self.userOptionsTb.setObjectName("userOptionsTb")
        self.userOptionsHL.addWidget(self.userOptionsTb, 0, QtCore.Qt.AlignRight)
        self.connectBtn = QtWidgets.QPushButton(self.MSColabConnectGb)
        self.connectBtn.setAutoDefault(True)
        self.connectBtn.setObjectName("connectBtn")
        self.userOptionsHL.addWidget(self.connectBtn)
        self.userOptionsHL.setStretch(0, 1)
        self.verticalLayout_3.addLayout(self.userOptionsHL)
        self.verticalLayout_2.addWidget(self.MSColabConnectGb)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.openFlightTracksGb = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openFlightTracksGb.sizePolicy().hasHeightForWidth())
        self.openFlightTracksGb.setSizePolicy(sizePolicy)
        self.openFlightTracksGb.setTitle("")
        self.openFlightTracksGb.setObjectName("openFlightTracksGb")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.openFlightTracksGb)
        self.verticalLayout.setContentsMargins(8, 8, 8, 8)
        self.verticalLayout.setObjectName("verticalLayout")
        self.openFlightTracksLabel = QtWidgets.QLabel(self.openFlightTracksGb)
        self.openFlightTracksLabel.setObjectName("openFlightTracksLabel")
        self.verticalLayout.addWidget(self.openFlightTracksLabel)
        self.listFlightTracks = QtWidgets.QListWidget(self.openFlightTracksGb)
        self.listFlightTracks.setFrameShadow(QtWidgets.QFrame.Plain)
        self.listFlightTracks.setObjectName("listFlightTracks")
        self.verticalLayout.addWidget(self.listFlightTracks)
        self.verticalLayout_5.addWidget(self.openFlightTracksGb)
        self.openViewsGb = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openViewsGb.sizePolicy().hasHeightForWidth())
        self.openViewsGb.setSizePolicy(sizePolicy)
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
        self.verticalLayout_5.addWidget(self.openViewsGb)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.openOperationsGb = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openOperationsGb.sizePolicy().hasHeightForWidth())
        self.openOperationsGb.setSizePolicy(sizePolicy)
        self.openOperationsGb.setTitle("")
        self.openOperationsGb.setObjectName("openOperationsGb")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.openOperationsGb)
        self.gridLayout_3.setContentsMargins(8, 8, 8, 8)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pbOpenOperationArchive = QtWidgets.QPushButton(self.openOperationsGb)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pbOpenOperationArchive.sizePolicy().hasHeightForWidth())
        self.pbOpenOperationArchive.setSizePolicy(sizePolicy)
        self.pbOpenOperationArchive.setObjectName("pbOpenOperationArchive")
        self.gridLayout_3.addWidget(self.pbOpenOperationArchive, 11, 0, 1, 2)
        self.workingStatusLabel = QtWidgets.QLabel(self.openOperationsGb)
        self.workingStatusLabel.setWordWrap(True)
        self.workingStatusLabel.setObjectName("workingStatusLabel")
        self.gridLayout_3.addWidget(self.workingStatusLabel, 6, 0, 1, 2)
        self.categoryLabel = QtWidgets.QLabel(self.openOperationsGb)
        self.categoryLabel.setObjectName("categoryLabel")
        self.gridLayout_3.addWidget(self.categoryLabel, 9, 0, 1, 1)
        self.workLocallyCheckbox = QtWidgets.QCheckBox(self.openOperationsGb)
        self.workLocallyCheckbox.setObjectName("workLocallyCheckbox")
        self.gridLayout_3.addWidget(self.workLocallyCheckbox, 10, 0, 1, 1)
        self.listOperationsMSC = QtWidgets.QListWidget(self.openOperationsGb)
        self.listOperationsMSC.setObjectName("listOperationsMSC")
        self.gridLayout_3.addWidget(self.listOperationsMSC, 4, 0, 1, 2)
        self.activeOperationDesc = QtWidgets.QLabel(self.openOperationsGb)
        self.activeOperationDesc.setLineWidth(1)
        self.activeOperationDesc.setObjectName("activeOperationDesc")
        self.gridLayout_3.addWidget(self.activeOperationDesc, 1, 0, 1, 1)
        self.activeOperationsLabel = QtWidgets.QLabel(self.openOperationsGb)
        self.activeOperationsLabel.setObjectName("activeOperationsLabel")
        self.gridLayout_3.addWidget(self.activeOperationsLabel, 2, 0, 1, 1)
        self.filterCategoryCb = QtWidgets.QComboBox(self.openOperationsGb)
        self.filterCategoryCb.setAutoFillBackground(False)
        self.filterCategoryCb.setEditable(False)
        self.filterCategoryCb.setObjectName("filterCategoryCb")
        self.filterCategoryCb.addItem("")
        self.gridLayout_3.addWidget(self.filterCategoryCb, 9, 1, 1, 1)
        self.serverOptionsCb = QtWidgets.QComboBox(self.openOperationsGb)
        self.serverOptionsCb.setObjectName("serverOptionsCb")
        self.serverOptionsCb.addItem("")
        self.serverOptionsCb.addItem("")
        self.serverOptionsCb.addItem("")
        self.gridLayout_3.addWidget(self.serverOptionsCb, 10, 1, 1, 1)
        self.userCountLabel = QtWidgets.QLabel(self.openOperationsGb)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.userCountLabel.sizePolicy().hasHeightForWidth())
        self.userCountLabel.setSizePolicy(sizePolicy)
        self.userCountLabel.setMinimumSize(QtCore.QSize(120, 0))
        self.userCountLabel.setMaximumSize(QtCore.QSize(120, 16777215))
        self.userCountLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.userCountLabel.setAutoFillBackground(False)
        self.userCountLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.userCountLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.userCountLabel.setScaledContents(False)
        self.userCountLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.userCountLabel.setObjectName("userCountLabel")
        self.gridLayout_3.addWidget(self.userCountLabel, 1, 1, 1, 1)
        self.horizontalLayout.addWidget(self.openOperationsGb)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 2)
        self.gridLayout.setColumnStretch(0, 1)
        MSUIMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MSUIMainWindow)
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
        MSUIMainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MSUIMainWindow)
        self.statusBar.setObjectName("statusBar")
        MSUIMainWindow.setStatusBar(self.statusBar)
        self.actionSaveActiveFlightTrack = QtWidgets.QAction(MSUIMainWindow)
        self.actionSaveActiveFlightTrack.setObjectName("actionSaveActiveFlightTrack")
        self.actionSaveActiveFlightTrackAs = QtWidgets.QAction(MSUIMainWindow)
        self.actionSaveActiveFlightTrackAs.setObjectName("actionSaveActiveFlightTrackAs")
        self.actionAboutMSUI = QtWidgets.QAction(MSUIMainWindow)
        self.actionAboutMSUI.setObjectName("actionAboutMSUI")
        self.actionOnlineHelp = QtWidgets.QAction(MSUIMainWindow)
        self.actionOnlineHelp.setObjectName("actionOnlineHelp")
        self.actionQuit = QtWidgets.QAction(MSUIMainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionShortcuts = QtWidgets.QAction(MSUIMainWindow)
        self.actionShortcuts.setObjectName("actionShortcuts")
        self.actionCloseSelectedFlightTrack = QtWidgets.QAction(MSUIMainWindow)
        self.actionCloseSelectedFlightTrack.setObjectName("actionCloseSelectedFlightTrack")
        self.actionUpdater = QtWidgets.QAction(MSUIMainWindow)
        self.actionUpdater.setObjectName("actionUpdater")
        self.actionConfiguration = QtWidgets.QAction(MSUIMainWindow)
        self.actionConfiguration.setObjectName("actionConfiguration")
        self.actionTopView = QtWidgets.QAction(MSUIMainWindow)
        self.actionTopView.setObjectName("actionTopView")
        self.actionSideView = QtWidgets.QAction(MSUIMainWindow)
        self.actionSideView.setObjectName("actionSideView")
        self.actionTableView = QtWidgets.QAction(MSUIMainWindow)
        self.actionTableView.setObjectName("actionTableView")
        self.actionLinearView = QtWidgets.QAction(MSUIMainWindow)
        self.actionLinearView.setObjectName("actionLinearView")
        self.actionChat = QtWidgets.QAction(MSUIMainWindow)
        self.actionChat.setObjectName("actionChat")
        self.actionVersionHistory = QtWidgets.QAction(MSUIMainWindow)
        self.actionVersionHistory.setObjectName("actionVersionHistory")
        self.actionManageUsers = QtWidgets.QAction(MSUIMainWindow)
        self.actionManageUsers.setObjectName("actionManageUsers")
        self.actionMSColabHelp = QtWidgets.QAction(MSUIMainWindow)
        self.actionMSColabHelp.setObjectName("actionMSColabHelp")
        self.actionDeleteOperation = QtWidgets.QAction(MSUIMainWindow)
        self.actionDeleteOperation.setObjectName("actionDeleteOperation")
        self.actionBringMainWindowToFront = QtWidgets.QAction(MSUIMainWindow)
        self.actionBringMainWindowToFront.setObjectName("actionBringMainWindowToFront")
        self.actionNewFlightTrack = QtWidgets.QAction(MSUIMainWindow)
        self.actionNewFlightTrack.setObjectName("actionNewFlightTrack")
        self.actionAddOperation = QtWidgets.QAction(MSUIMainWindow)
        self.actionAddOperation.setObjectName("actionAddOperation")
        self.actionSearch = QtWidgets.QAction(MSUIMainWindow)
        self.actionSearch.setObjectName("actionSearch")
        self.actionViewDescription = QtWidgets.QAction(MSUIMainWindow)
        self.actionViewDescription.setObjectName("actionViewDescription")
        self.actionChangeDescription = QtWidgets.QAction(MSUIMainWindow)
        self.actionChangeDescription.setObjectName("actionChangeDescription")
        self.actionRenameOperation = QtWidgets.QAction(MSUIMainWindow)
        self.actionRenameOperation.setObjectName("actionRenameOperation")
        self.actionLeaveOperation = QtWidgets.QAction(MSUIMainWindow)
        self.actionLeaveOperation.setObjectName("actionLeaveOperation")
        self.actionArchiveOperation = QtWidgets.QAction(MSUIMainWindow)
        self.actionArchiveOperation.setObjectName("actionArchiveOperation")
        self.actionChangeCategory = QtWidgets.QAction(MSUIMainWindow)
        self.actionChangeCategory.setObjectName("actionChangeCategory")
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
        self.menuProperties.addAction(self.actionChangeCategory)
        self.menuProperties.addAction(self.actionChangeDescription)
        self.menuProperties.addAction(self.actionManageUsers)
        self.menuProperties.addAction(self.actionRenameOperation)
        self.menuProperties.addAction(self.actionDeleteOperation)
        self.menuProperties.addAction(self.actionArchiveOperation)
        self.menuOperation.addAction(self.actionChat)
        self.menuOperation.addAction(self.actionVersionHistory)
        self.menuOperation.addAction(self.actionViewDescription)
        self.menuOperation.addSeparator()
        self.menuOperation.addAction(self.actionLeaveOperation)
        self.menuOperation.addAction(self.menuProperties.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuViews.menuAction())
        self.menubar.addAction(self.menuOperation.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MSUIMainWindow)
        self.filterCategoryCb.setCurrentIndex(0)
        self.actionQuit.triggered.connect(MSUIMainWindow.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MSUIMainWindow)
        MSUIMainWindow.setTabOrder(self.connectBtn, self.userOptionsTb)
        MSUIMainWindow.setTabOrder(self.userOptionsTb, self.listFlightTracks)
        MSUIMainWindow.setTabOrder(self.listFlightTracks, self.listViews)
        MSUIMainWindow.setTabOrder(self.listViews, self.listOperationsMSC)
        MSUIMainWindow.setTabOrder(self.listOperationsMSC, self.workLocallyCheckbox)
        MSUIMainWindow.setTabOrder(self.workLocallyCheckbox, self.serverOptionsCb)

    def retranslateUi(self, MSUIMainWindow):
        _translate = QtCore.QCoreApplication.translate
        MSUIMainWindow.setWindowTitle(_translate("MSUIMainWindow", "Mission Support System"))
        self.mscolabServerLabel.setText(_translate("MSUIMainWindow", "MSColab Server"))
        self.mscStatusLabel.setText(_translate("MSUIMainWindow", "Status: disconnected"))
        self.usernameLabel.setText(_translate("MSUIMainWindow", "User"))
        self.userOptionsTb.setToolTip(_translate("MSUIMainWindow", "Profile options"))
        self.connectBtn.setToolTip(_translate("MSUIMainWindow", "Connect to an MSColab Server"))
        self.connectBtn.setText(_translate("MSUIMainWindow", "Connect"))
        self.openFlightTracksLabel.setText(_translate("MSUIMainWindow", "Flight Tracks:"))
        self.listFlightTracks.setToolTip(_translate("MSUIMainWindow", "List of open flight tracks.\n"
"Double-click a flight track to activate it.\n"
"Save a flight track to name it."))
        self.listFlightTracks.setSortingEnabled(False)
        self.openViewsLabel.setText(_translate("MSUIMainWindow", "Open Views:"))
        self.listViews.setToolTip(_translate("MSUIMainWindow", "Double-click a view to bring it to the front."))
        self.pbOpenOperationArchive.setText(_translate("MSUIMainWindow", "Operation Archive"))
        self.workingStatusLabel.setText(_translate("MSUIMainWindow", "No operations selected"))
        self.categoryLabel.setText(_translate("MSUIMainWindow", "Category:"))
        self.workLocallyCheckbox.setToolTip(_translate("MSUIMainWindow", "Check to work asynchronously from the server"))
        self.workLocallyCheckbox.setText(_translate("MSUIMainWindow", "Work Asynchronously"))
        self.listOperationsMSC.setToolTip(_translate("MSUIMainWindow", "List of mscolab operations.\n"
"Double click a operation to activate and view its description."))
        self.activeOperationDesc.setText(_translate("MSUIMainWindow", "Select Operation to View Description"))
        self.activeOperationsLabel.setText(_translate("MSUIMainWindow", "Operations"))
        self.filterCategoryCb.setWhatsThis(_translate("MSUIMainWindow", "filter by operation category"))
        self.filterCategoryCb.setCurrentText(_translate("MSUIMainWindow", "ANY"))
        self.filterCategoryCb.setItemText(0, _translate("MSUIMainWindow", "ANY"))
        self.serverOptionsCb.setToolTip(_translate("MSUIMainWindow", "Fetch/Save Server options"))
        self.serverOptionsCb.setItemText(0, _translate("MSUIMainWindow", "Server Options"))
        self.serverOptionsCb.setItemText(1, _translate("MSUIMainWindow", "Fetch From Server"))
        self.serverOptionsCb.setItemText(2, _translate("MSUIMainWindow", "Save To Server"))
        self.userCountLabel.setText(_translate("MSUIMainWindow", "Active Users: 0"))
        self.menuFile.setTitle(_translate("MSUIMainWindow", "&File"))
        self.menuImportFlightTrack.setTitle(_translate("MSUIMainWindow", "Import Flight Track"))
        self.menuExportActiveFlightTrack.setTitle(_translate("MSUIMainWindow", "Export Flight Track"))
        self.menuNew.setTitle(_translate("MSUIMainWindow", "New"))
        self.menuHelp.setTitle(_translate("MSUIMainWindow", "&Help"))
        self.menuViews.setTitle(_translate("MSUIMainWindow", "Views"))
        self.menuOperation.setTitle(_translate("MSUIMainWindow", "Operation"))
        self.menuProperties.setTitle(_translate("MSUIMainWindow", "Maintenance"))
        self.actionSaveActiveFlightTrack.setText(_translate("MSUIMainWindow", "&Save Active Flight Track"))
        self.actionSaveActiveFlightTrack.setShortcut(_translate("MSUIMainWindow", "Ctrl+S"))
        self.actionSaveActiveFlightTrackAs.setText(_translate("MSUIMainWindow", "Save Active Flight Track As"))
        self.actionSaveActiveFlightTrackAs.setShortcut(_translate("MSUIMainWindow", "Ctrl+Shift+S"))
        self.actionAboutMSUI.setText(_translate("MSUIMainWindow", "&About MSUI"))
        self.actionOnlineHelp.setText(_translate("MSUIMainWindow", "&Online Help"))
        self.actionQuit.setText(_translate("MSUIMainWindow", "&Quit"))
        self.actionQuit.setShortcut(_translate("MSUIMainWindow", "Ctrl+Q"))
        self.actionShortcuts.setText(_translate("MSUIMainWindow", "&Shortcuts"))
        self.actionShortcuts.setToolTip(_translate("MSUIMainWindow", "Open the shortcuts dialog"))
        self.actionShortcuts.setShortcut(_translate("MSUIMainWindow", "Alt+S"))
        self.actionCloseSelectedFlightTrack.setText(_translate("MSUIMainWindow", "&Close Selected Local Flight Track"))
        self.actionUpdater.setText(_translate("MSUIMainWindow", "&Updater"))
        self.actionConfiguration.setText(_translate("MSUIMainWindow", "&Configuration"))
        self.actionTopView.setText(_translate("MSUIMainWindow", "&Top View (Horizontal Section)"))
        self.actionTopView.setShortcut(_translate("MSUIMainWindow", "Ctrl+H"))
        self.actionSideView.setText(_translate("MSUIMainWindow", "&Side View (Vertical Section)"))
        self.actionSideView.setShortcut(_translate("MSUIMainWindow", "Ctrl+V"))
        self.actionTableView.setText(_translate("MSUIMainWindow", "&Table View"))
        self.actionTableView.setShortcut(_translate("MSUIMainWindow", "Ctrl+T"))
        self.actionLinearView.setText(_translate("MSUIMainWindow", "&Linear View"))
        self.actionLinearView.setShortcut(_translate("MSUIMainWindow", "Ctrl+L"))
        self.actionChat.setText(_translate("MSUIMainWindow", "&Chat"))
        self.actionVersionHistory.setText(_translate("MSUIMainWindow", "&Version History"))
        self.actionManageUsers.setText(_translate("MSUIMainWindow", "&Manage Users"))
        self.actionMSColabHelp.setText(_translate("MSUIMainWindow", "&MSColab"))
        self.actionDeleteOperation.setText(_translate("MSUIMainWindow", "&Delete Operation"))
        self.actionBringMainWindowToFront.setText(_translate("MSUIMainWindow", "Bring Main Window To Front"))
        self.actionBringMainWindowToFront.setShortcut(_translate("MSUIMainWindow", "Ctrl+Up"))
        self.actionNewFlightTrack.setText(_translate("MSUIMainWindow", "&Local Flight Track"))
        self.actionAddOperation.setText(_translate("MSUIMainWindow", "&MSColab Operation"))
        self.actionSearch.setText(_translate("MSUIMainWindow", "Search"))
        self.actionSearch.setToolTip(_translate("MSUIMainWindow", "Search for interactive text in the UI"))
        self.actionSearch.setShortcut(_translate("MSUIMainWindow", "Ctrl+F"))
        self.actionViewDescription.setText(_translate("MSUIMainWindow", "View Description"))
        self.actionChangeDescription.setText(_translate("MSUIMainWindow", "Change Description"))
        self.actionRenameOperation.setText(_translate("MSUIMainWindow", "Rename Operation"))
        self.actionLeaveOperation.setText(_translate("MSUIMainWindow", "&Leave Operation"))
        self.actionArchiveOperation.setText(_translate("MSUIMainWindow", "Archive Operation"))
        self.actionChangeCategory.setText(_translate("MSUIMainWindow", "Change Category"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MSUIMainWindow = QtWidgets.QMainWindow()
    ui = Ui_MSUIMainWindow()
    ui.setupUi(MSUIMainWindow)
    MSUIMainWindow.show()
    sys.exit(app.exec_())
