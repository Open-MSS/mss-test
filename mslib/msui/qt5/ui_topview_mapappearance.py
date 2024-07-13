# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/zandar/Desktop/MSS/mslib/msui/ui/ui_topview_mapappearance.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MapAppearanceDialog(object):
    def setupUi(self, MapAppearanceDialog):
        MapAppearanceDialog.setObjectName("MapAppearanceDialog")
        MapAppearanceDialog.resize(394, 465)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MapAppearanceDialog.sizePolicy().hasHeightForWidth())
        MapAppearanceDialog.setSizePolicy(sizePolicy)
        self.verticalLayout = QtWidgets.QVBoxLayout(MapAppearanceDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.cbDrawGraticule = QtWidgets.QCheckBox(MapAppearanceDialog)
        self.cbDrawGraticule.setObjectName("cbDrawGraticule")
        self.verticalLayout.addWidget(self.cbDrawGraticule)
        self.cbDrawCoastlines = QtWidgets.QCheckBox(MapAppearanceDialog)
        self.cbDrawCoastlines.setChecked(True)
        self.cbDrawCoastlines.setObjectName("cbDrawCoastlines")
        self.verticalLayout.addWidget(self.cbDrawCoastlines)
        self.cbLabelFlightTrack = QtWidgets.QCheckBox(MapAppearanceDialog)
        self.cbLabelFlightTrack.setObjectName("cbLabelFlightTrack")
        self.verticalLayout.addWidget(self.cbLabelFlightTrack)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cbFillWaterBodies = QtWidgets.QCheckBox(MapAppearanceDialog)
        self.cbFillWaterBodies.setEnabled(True)
        self.cbFillWaterBodies.setMinimumSize(QtCore.QSize(145, 0))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18, 128))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18, 128))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(173, 173, 173))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18, 128))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.cbFillWaterBodies.setPalette(palette)
        self.cbFillWaterBodies.setChecked(True)
        self.cbFillWaterBodies.setObjectName("cbFillWaterBodies")
        self.horizontalLayout.addWidget(self.cbFillWaterBodies)
        self.btWaterColour = QtWidgets.QPushButton(MapAppearanceDialog)
        self.btWaterColour.setMinimumSize(QtCore.QSize(135, 0))
        self.btWaterColour.setFlat(False)
        self.btWaterColour.setObjectName("btWaterColour")
        self.horizontalLayout.addWidget(self.btWaterColour)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.cbFillContinents = QtWidgets.QCheckBox(MapAppearanceDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbFillContinents.sizePolicy().hasHeightForWidth())
        self.cbFillContinents.setSizePolicy(sizePolicy)
        self.cbFillContinents.setMinimumSize(QtCore.QSize(145, 0))
        self.cbFillContinents.setObjectName("cbFillContinents")
        self.horizontalLayout_2.addWidget(self.cbFillContinents)
        self.btLandColour = QtWidgets.QPushButton(MapAppearanceDialog)
        self.btLandColour.setMinimumSize(QtCore.QSize(135, 0))
        self.btLandColour.setObjectName("btLandColour")
        self.horizontalLayout_2.addWidget(self.btLandColour)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.groupBox_2 = QtWidgets.QGroupBox(MapAppearanceDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(150)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName("groupBox_2")
        self.cbLineStyle = QtWidgets.QComboBox(self.groupBox_2)
        self.cbLineStyle.setGeometry(QtCore.QRect(170, 120, 131, 22))
        self.cbLineStyle.setObjectName("cbLineStyle")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(10, 120, 61, 16))
        self.label_4.setObjectName("label_4")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 150, 91, 16))
        self.label_2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.label_2.setObjectName("label_2")
        self.sbLineThickness = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.sbLineThickness.setGeometry(QtCore.QRect(170, 150, 131, 21))
        self.sbLineThickness.setObjectName("sbLineThickness")
        self.cbDrawMarker = QtWidgets.QCheckBox(self.groupBox_2)
        self.cbDrawMarker.setGeometry(QtCore.QRect(10, 30, 145, 21))
        self.cbDrawMarker.setMinimumSize(QtCore.QSize(145, 0))
        self.cbDrawMarker.setObjectName("cbDrawMarker")
        self.btWaypointsColour = QtWidgets.QPushButton(self.groupBox_2)
        self.btWaypointsColour.setGeometry(QtCore.QRect(170, 30, 135, 23))
        self.btWaypointsColour.setMinimumSize(QtCore.QSize(135, 0))
        self.btWaypointsColour.setObjectName("btWaypointsColour")
        self.cbDrawFlightTrack = QtWidgets.QCheckBox(self.groupBox_2)
        self.cbDrawFlightTrack.setEnabled(True)
        self.cbDrawFlightTrack.setGeometry(QtCore.QRect(10, 60, 145, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbDrawFlightTrack.sizePolicy().hasHeightForWidth())
        self.cbDrawFlightTrack.setSizePolicy(sizePolicy)
        self.cbDrawFlightTrack.setMinimumSize(QtCore.QSize(145, 0))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(20, 19, 18))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(173, 173, 173))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        self.cbDrawFlightTrack.setPalette(palette)
        self.cbDrawFlightTrack.setChecked(True)
        self.cbDrawFlightTrack.setObjectName("cbDrawFlightTrack")
        self.btVerticesColour = QtWidgets.QPushButton(self.groupBox_2)
        self.btVerticesColour.setGeometry(QtCore.QRect(170, 60, 135, 23))
        self.btVerticesColour.setMinimumSize(QtCore.QSize(135, 0))
        self.btVerticesColour.setObjectName("btVerticesColour")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(10, 90, 131, 16))
        self.label_3.setObjectName("label_3")
        self.hsTransparencyControl = QtWidgets.QSlider(self.groupBox_2)
        self.hsTransparencyControl.setGeometry(QtCore.QRect(170, 90, 131, 22))
        self.hsTransparencyControl.setProperty("value", 20)
        self.hsTransparencyControl.setOrientation(QtCore.Qt.Horizontal)
        self.hsTransparencyControl.setObjectName("hsTransparencyControl")
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(MapAppearanceDialog)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 3, 2, 1, 1)
        self.tov_cbtitlesize = QtWidgets.QComboBox(self.groupBox)
        self.tov_cbtitlesize.setObjectName("tov_cbtitlesize")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.tov_cbtitlesize.addItem("")
        self.gridLayout_4.addWidget(self.tov_cbtitlesize, 3, 1, 1, 1)
        self.tov_cbaxessize = QtWidgets.QComboBox(self.groupBox)
        self.tov_cbaxessize.setObjectName("tov_cbaxessize")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.tov_cbaxessize.addItem("")
        self.gridLayout_4.addWidget(self.tov_cbaxessize, 3, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 3, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.buttonBox = QtWidgets.QDialogButtonBox(MapAppearanceDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(MapAppearanceDialog)
        self.buttonBox.accepted.connect(MapAppearanceDialog.accept)
        self.buttonBox.rejected.connect(MapAppearanceDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(MapAppearanceDialog)

    def retranslateUi(self, MapAppearanceDialog):
        _translate = QtCore.QCoreApplication.translate
        MapAppearanceDialog.setWindowTitle(_translate("MapAppearanceDialog", "Top View Map Appearance"))
        self.cbDrawGraticule.setText(_translate("MapAppearanceDialog", "draw graticule"))
        self.cbDrawCoastlines.setText(_translate("MapAppearanceDialog", "draw coastlines"))
        self.cbLabelFlightTrack.setText(_translate("MapAppearanceDialog", "label flight path"))
        self.cbFillWaterBodies.setText(_translate("MapAppearanceDialog", "fill map background"))
        self.btWaterColour.setText(_translate("MapAppearanceDialog", "colour"))
        self.cbFillContinents.setText(_translate("MapAppearanceDialog", "fill continents"))
        self.btLandColour.setText(_translate("MapAppearanceDialog", "colour"))
        self.groupBox_2.setTitle(_translate("MapAppearanceDialog", "Flight Track style options"))
        self.label_4.setText(_translate("MapAppearanceDialog", "Line style"))
        self.label_2.setText(_translate("MapAppearanceDialog", "Line Thickness"))
        self.cbDrawMarker.setToolTip(_translate("MapAppearanceDialog", "Draw a circle marker on every waypoint along the flight track"))
        self.cbDrawMarker.setText(_translate("MapAppearanceDialog", "draw marker"))
        self.btWaypointsColour.setText(_translate("MapAppearanceDialog", "colour of waypoints"))
        self.cbDrawFlightTrack.setText(_translate("MapAppearanceDialog", "draw flight path"))
        self.btVerticesColour.setText(_translate("MapAppearanceDialog", "colour of flight path"))
        self.label_3.setText(_translate("MapAppearanceDialog", "Transparency control"))
        self.groupBox.setTitle(_translate("MapAppearanceDialog", "Plot Options"))
        self.label_7.setText(_translate("MapAppearanceDialog", " Axes Label Size  "))
        self.tov_cbtitlesize.setItemText(0, _translate("MapAppearanceDialog", "default"))
        self.tov_cbtitlesize.setItemText(1, _translate("MapAppearanceDialog", "4"))
        self.tov_cbtitlesize.setItemText(2, _translate("MapAppearanceDialog", "6"))
        self.tov_cbtitlesize.setItemText(3, _translate("MapAppearanceDialog", "8"))
        self.tov_cbtitlesize.setItemText(4, _translate("MapAppearanceDialog", "10"))
        self.tov_cbtitlesize.setItemText(5, _translate("MapAppearanceDialog", "12"))
        self.tov_cbtitlesize.setItemText(6, _translate("MapAppearanceDialog", "14"))
        self.tov_cbtitlesize.setItemText(7, _translate("MapAppearanceDialog", "16"))
        self.tov_cbtitlesize.setItemText(8, _translate("MapAppearanceDialog", "18"))
        self.tov_cbtitlesize.setItemText(9, _translate("MapAppearanceDialog", "20"))
        self.tov_cbtitlesize.setItemText(10, _translate("MapAppearanceDialog", "22"))
        self.tov_cbtitlesize.setItemText(11, _translate("MapAppearanceDialog", "24"))
        self.tov_cbtitlesize.setItemText(12, _translate("MapAppearanceDialog", "26"))
        self.tov_cbtitlesize.setItemText(13, _translate("MapAppearanceDialog", "28"))
        self.tov_cbtitlesize.setItemText(14, _translate("MapAppearanceDialog", "30"))
        self.tov_cbtitlesize.setItemText(15, _translate("MapAppearanceDialog", "32"))
        self.tov_cbaxessize.setItemText(0, _translate("MapAppearanceDialog", "default"))
        self.tov_cbaxessize.setItemText(1, _translate("MapAppearanceDialog", "4"))
        self.tov_cbaxessize.setItemText(2, _translate("MapAppearanceDialog", "6"))
        self.tov_cbaxessize.setItemText(3, _translate("MapAppearanceDialog", "8"))
        self.tov_cbaxessize.setItemText(4, _translate("MapAppearanceDialog", "10"))
        self.tov_cbaxessize.setItemText(5, _translate("MapAppearanceDialog", "12"))
        self.tov_cbaxessize.setItemText(6, _translate("MapAppearanceDialog", "14"))
        self.tov_cbaxessize.setItemText(7, _translate("MapAppearanceDialog", "16"))
        self.tov_cbaxessize.setItemText(8, _translate("MapAppearanceDialog", "18"))
        self.tov_cbaxessize.setItemText(9, _translate("MapAppearanceDialog", "20"))
        self.tov_cbaxessize.setItemText(10, _translate("MapAppearanceDialog", "22"))
        self.tov_cbaxessize.setItemText(11, _translate("MapAppearanceDialog", "24"))
        self.tov_cbaxessize.setItemText(12, _translate("MapAppearanceDialog", "26"))
        self.tov_cbaxessize.setItemText(13, _translate("MapAppearanceDialog", "28"))
        self.tov_cbaxessize.setItemText(14, _translate("MapAppearanceDialog", "30"))
        self.tov_cbaxessize.setItemText(15, _translate("MapAppearanceDialog", "32"))
        self.label.setText(_translate("MapAppearanceDialog", " Plot Title Size"))
