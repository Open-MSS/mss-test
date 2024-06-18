# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_topview_mapappearance.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MapAppearanceDialog(object):
    def setupUi(self, MapAppearanceDialog):
        MapAppearanceDialog.setObjectName("MapAppearanceDialog")
        MapAppearanceDialog.resize(394, 489)
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
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.cbDrawMarker = QtWidgets.QCheckBox(MapAppearanceDialog)
        self.cbDrawMarker.setMinimumSize(QtCore.QSize(145, 0))
        self.cbDrawMarker.setObjectName("cbDrawMarker")
        self.horizontalLayout_5.addWidget(self.cbDrawMarker)
        self.btWaypointsColour = QtWidgets.QPushButton(MapAppearanceDialog)
        self.btWaypointsColour.setMinimumSize(QtCore.QSize(135, 0))
        self.btWaypointsColour.setObjectName("btWaypointsColour")
        self.horizontalLayout_5.addWidget(self.btWaypointsColour)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.cbDrawFlightTrack = QtWidgets.QCheckBox(MapAppearanceDialog)
        self.cbDrawFlightTrack.setEnabled(True)
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
        self.horizontalLayout_3.addWidget(self.cbDrawFlightTrack)
        self.btVerticesColour = QtWidgets.QPushButton(MapAppearanceDialog)
        self.btVerticesColour.setMinimumSize(QtCore.QSize(135, 0))
        self.btVerticesColour.setObjectName("btVerticesColour")
        self.horizontalLayout_3.addWidget(self.btVerticesColour)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.frame = QtWidgets.QFrame(MapAppearanceDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(100)
        sizePolicy.setVerticalStretch(200)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalSlider = QtWidgets.QSlider(self.frame)
        self.horizontalSlider.setGeometry(QtCore.QRect(10, 90, 160, 22))
        self.horizontalSlider.setProperty("value", 20)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(190, 90, 131, 16))
        self.label_3.setObjectName("label_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 0, 361, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setGeometry(QtCore.QRect(10, 60, 161, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(190, 60, 61, 16))
        self.label_4.setObjectName("label_4")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(110, 30, 91, 16))
        self.label_2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.label_2.setObjectName("label_2")
        self.sbLineThickness = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.sbLineThickness.setGeometry(QtCore.QRect(10, 20, 81, 31))
        self.sbLineThickness.setObjectName("sbLineThickness")
        self.groupBox_2.raise_()
        self.horizontalSlider.raise_()
        self.label_3.raise_()
        self.verticalLayout.addWidget(self.frame)
        self.groupBox = QtWidgets.QGroupBox(MapAppearanceDialog)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
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
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 3, 2, 1, 1)
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
        self.cbDrawMarker.setToolTip(_translate("MapAppearanceDialog", "Draw a circle marker on every waypoint along the flight track"))
        self.cbDrawMarker.setText(_translate("MapAppearanceDialog", "draw marker"))
        self.btWaypointsColour.setText(_translate("MapAppearanceDialog", "colour of waypoints"))
        self.cbDrawFlightTrack.setText(_translate("MapAppearanceDialog", "draw flight path"))
        self.btVerticesColour.setText(_translate("MapAppearanceDialog", "colour of flight path"))
        self.label_3.setText(_translate("MapAppearanceDialog", "Transparency control"))
        self.groupBox_2.setTitle(_translate("MapAppearanceDialog", "Flight path option"))
        self.comboBox.setItemText(0, _translate("MapAppearanceDialog", "default (solid line)"))
        self.comboBox.setItemText(1, _translate("MapAppearanceDialog", "dashed line"))
        self.comboBox.setItemText(2, _translate("MapAppearanceDialog", "dotted"))
        self.comboBox.setItemText(3, _translate("MapAppearanceDialog", "dashdot"))
        self.label_4.setText(_translate("MapAppearanceDialog", "Line style"))
        self.label_2.setText(_translate("MapAppearanceDialog", "Line Thickness"))
        self.groupBox.setTitle(_translate("MapAppearanceDialog", "Plot Options"))
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
        self.label_7.setText(_translate("MapAppearanceDialog", " Axes Label Size  "))
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
