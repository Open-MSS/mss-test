# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mssau.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(900, 300)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.stimeLabel = QtWidgets.QLabel(Form)
        self.stimeLabel.setObjectName("stimeLabel")
        self.verticalLayout_2.addWidget(self.stimeLabel)
        self.stimeSpinBox = QtWidgets.QDateTimeEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stimeSpinBox.sizePolicy().hasHeightForWidth())
        self.stimeSpinBox.setSizePolicy(sizePolicy)
        self.stimeSpinBox.setMinimumSize(QtCore.QSize(160, 0))
        self.stimeSpinBox.setMaximumSize(QtCore.QSize(300, 16777215))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.stimeSpinBox.setFont(font)
        self.stimeSpinBox.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToPreviousValue)
        self.stimeSpinBox.setCalendarPopup(False)
        self.stimeSpinBox.setTimeSpec(QtCore.Qt.UTC)
        self.stimeSpinBox.setObjectName("stimeSpinBox")
        self.verticalLayout_2.addWidget(self.stimeSpinBox)
        self.horizontalLayout_9.addLayout(self.verticalLayout_2)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.etimeLabel = QtWidgets.QLabel(Form)
        self.etimeLabel.setObjectName("etimeLabel")
        self.verticalLayout_8.addWidget(self.etimeLabel)
        self.etimeSpinBox = QtWidgets.QDateTimeEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.etimeSpinBox.sizePolicy().hasHeightForWidth())
        self.etimeSpinBox.setSizePolicy(sizePolicy)
        self.etimeSpinBox.setMinimumSize(QtCore.QSize(160, 0))
        self.etimeSpinBox.setMaximumSize(QtCore.QSize(300, 16777215))
        self.etimeSpinBox.setDate(QtCore.QDate(2009, 12, 17))
        self.etimeSpinBox.setMinimumDateTime(QtCore.QDateTime(QtCore.QDate(1999, 12, 29), QtCore.QTime(12, 0, 0)))
        self.etimeSpinBox.setMinimumDate(QtCore.QDate(1999, 12, 29))
        self.etimeSpinBox.setMinimumTime(QtCore.QTime(12, 0, 0))
        self.etimeSpinBox.setCalendarPopup(False)
        self.etimeSpinBox.setTimeSpec(QtCore.Qt.UTC)
        self.etimeSpinBox.setObjectName("etimeSpinBox")
        self.verticalLayout_8.addWidget(self.etimeSpinBox)
        self.horizontalLayout_9.addLayout(self.verticalLayout_8)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.timeIntervalLabel = QtWidgets.QLabel(Form)
        self.timeIntervalLabel.setObjectName("timeIntervalLabel")
        self.verticalLayout_10.addWidget(self.timeIntervalLabel)
        self.timeIntervalComboBox = QtWidgets.QComboBox(Form)
        self.timeIntervalComboBox.setObjectName("timeIntervalComboBox")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.setItemText(0, "")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.timeIntervalComboBox.addItem("")
        self.verticalLayout_10.addWidget(self.timeIntervalComboBox)
        self.horizontalLayout_9.addLayout(self.verticalLayout_10)
        self.gridLayout_2.addLayout(self.horizontalLayout_9, 3, 0, 1, 1)
        self.selectConfigButton = QtWidgets.QPushButton(Form)
        self.selectConfigButton.setObjectName("selectConfigButton")
        self.gridLayout_2.addWidget(self.selectConfigButton, 0, 0, 1, 1)
        self.autoplotSecsTreeWidget = QtWidgets.QTreeWidget(Form)
        self.autoplotSecsTreeWidget.setMinimumSize(QtCore.QSize(0, 100))
        self.autoplotSecsTreeWidget.setMaximumSize(QtCore.QSize(16777215, 200))
        self.autoplotSecsTreeWidget.setObjectName("autoplotSecsTreeWidget")
        self.autoplotSecsTreeWidget.header().setDefaultSectionSize(200)
        self.gridLayout_2.addWidget(self.autoplotSecsTreeWidget, 1, 1, 1, 1)
        self.autoplotTreeWidget = QtWidgets.QTreeWidget(Form)
        self.autoplotTreeWidget.setMinimumSize(QtCore.QSize(0, 100))
        self.autoplotTreeWidget.setMaximumSize(QtCore.QSize(16777215, 200))
        self.autoplotTreeWidget.setObjectName("autoplotTreeWidget")
        self.autoplotTreeWidget.header().setDefaultSectionSize(150)
        self.gridLayout_2.addWidget(self.autoplotTreeWidget, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(250, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.updateConfigFile = QtWidgets.QPushButton(Form)
        self.updateConfigFile.setObjectName("updateConfigFile")
        self.horizontalLayout_2.addWidget(self.updateConfigFile)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 3, 1, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.addToAutoplotButton = QtWidgets.QPushButton(Form)
        self.addToAutoplotButton.setObjectName("addToAutoplotButton")
        self.horizontalLayout_4.addWidget(self.addToAutoplotButton)
        self.RemoveFromAutoplotButton = QtWidgets.QPushButton(Form)
        self.RemoveFromAutoplotButton.setObjectName("RemoveFromAutoplotButton")
        self.horizontalLayout_4.addWidget(self.RemoveFromAutoplotButton)
        self.UploadAutoplotButton = QtWidgets.QPushButton(Form)
        self.UploadAutoplotButton.setObjectName("UploadAutoplotButton")
        self.horizontalLayout_4.addWidget(self.UploadAutoplotButton)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 2, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.addToAutoplotSecsButton = QtWidgets.QPushButton(Form)
        self.addToAutoplotSecsButton.setObjectName("addToAutoplotSecsButton")
        self.horizontalLayout_5.addWidget(self.addToAutoplotSecsButton)
        self.RemoveFromAutoplotSecsButton = QtWidgets.QPushButton(Form)
        self.RemoveFromAutoplotSecsButton.setObjectName("RemoveFromAutoplotSecsButton")
        self.horizontalLayout_5.addWidget(self.RemoveFromAutoplotSecsButton)
        self.UploadAutoplotSecsButton = QtWidgets.QPushButton(Form)
        self.UploadAutoplotSecsButton.setObjectName("UploadAutoplotSecsButton")
        self.horizontalLayout_5.addWidget(self.UploadAutoplotSecsButton)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 2, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "MSSAUTOPLOT"))
        self.stimeLabel.setText(_translate("Form", "Start Time"))
        self.stimeSpinBox.setToolTip(_translate("Form", "You can also specify an initialisation date here."))
        self.stimeSpinBox.setDisplayFormat(_translate("Form", "yyyy/MM/dd hh:mm UTC"))
        self.etimeLabel.setText(_translate("Form", "End Time"))
        self.etimeSpinBox.setToolTip(_translate("Form", "Specify the time value here, especially if the server does not provide predefined values. Keep in mind that the specified value may not be available from the server, though."))
        self.etimeSpinBox.setDisplayFormat(_translate("Form", "yyyy/MM/dd hh:mm UTC"))
        self.timeIntervalLabel.setText(_translate("Form", "Time Interval"))
        self.timeIntervalComboBox.setItemText(1, _translate("Form", "1 min"))
        self.timeIntervalComboBox.setItemText(2, _translate("Form", "5 min"))
        self.timeIntervalComboBox.setItemText(3, _translate("Form", "10 min"))
        self.timeIntervalComboBox.setItemText(4, _translate("Form", "15 min"))
        self.timeIntervalComboBox.setItemText(5, _translate("Form", "30 min"))
        self.timeIntervalComboBox.setItemText(6, _translate("Form", "1 hour"))
        self.timeIntervalComboBox.setItemText(7, _translate("Form", "2 hours"))
        self.timeIntervalComboBox.setItemText(8, _translate("Form", "3 hours"))
        self.timeIntervalComboBox.setItemText(9, _translate("Form", "6 hours"))
        self.timeIntervalComboBox.setItemText(10, _translate("Form", "12 hours"))
        self.timeIntervalComboBox.setItemText(11, _translate("Form", "24 hours"))
        self.timeIntervalComboBox.setItemText(12, _translate("Form", "2 days"))
        self.timeIntervalComboBox.setItemText(13, _translate("Form", "7 days"))
        self.selectConfigButton.setText(_translate("Form", "Select Configuration File"))
        self.autoplotSecsTreeWidget.headerItem().setText(0, _translate("Form", "URL"))
        self.autoplotSecsTreeWidget.headerItem().setText(1, _translate("Form", "Layers"))
        self.autoplotSecsTreeWidget.headerItem().setText(2, _translate("Form", "Styles"))
        self.autoplotSecsTreeWidget.headerItem().setText(3, _translate("Form", "Levels"))
        self.autoplotSecsTreeWidget.headerItem().setText(4, _translate("Form", "Start Time"))
        self.autoplotSecsTreeWidget.headerItem().setText(5, _translate("Form", "End Time"))
        self.autoplotSecsTreeWidget.headerItem().setText(6, _translate("Form", "Time Interval"))
        self.autoplotTreeWidget.headerItem().setText(0, _translate("Form", "Flight"))
        self.autoplotTreeWidget.headerItem().setText(1, _translate("Form", "Map Sections"))
        self.autoplotTreeWidget.headerItem().setText(2, _translate("Form", "Vertical"))
        self.autoplotTreeWidget.headerItem().setText(3, _translate("Form", "Filename"))
        self.autoplotTreeWidget.headerItem().setText(4, _translate("Form", "Initial Time"))
        self.autoplotTreeWidget.headerItem().setText(5, _translate("Form", "Valid Time"))
        self.updateConfigFile.setText(_translate("Form", "Update/Create Configuration File"))
        self.addToAutoplotButton.setText(_translate("Form", "Add"))
        self.RemoveFromAutoplotButton.setText(_translate("Form", "Remove"))
        self.UploadAutoplotButton.setText(_translate("Form", "Update"))
        self.addToAutoplotSecsButton.setText(_translate("Form", "Add"))
        self.RemoveFromAutoplotSecsButton.setText(_translate("Form", "Remove"))
        self.UploadAutoplotSecsButton.setText(_translate("Form", "Update"))
