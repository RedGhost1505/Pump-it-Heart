# Form implementation generated from reading ui file 'PantallaMuestra.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_SecondaryWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1380, 800)
        MainWindow.setMinimumSize(QtCore.QSize(1380, 800))
        MainWindow.setMaximumSize(QtCore.QSize(1380, 800))
        MainWindow.setStyleSheet("background-color:rgb(0, 0, 0)")
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(580, 60, 186, 64))
        font = QtGui.QFont()
        font.setFamily("Montserrat ExtraBold")
        font.setPointSize(39)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("color:rgb(255, 255, 255)")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(770, 60, 40, 64))
        font = QtGui.QFont()
        font.setFamily("Montserrat ExtraBold")
        font.setPointSize(39)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setStyleSheet("color:rgb(226,33, 28)")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 100, 103, 41))
        font = QtGui.QFont()
        font.setFamily("Montserrat ExtraBold")
        font.setPointSize(25)
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setAutoFillBackground(False)
        self.label_3.setStyleSheet("color:rgb(160,160, 160)")
        self.label_3.setObjectName("label_3")
        self.imageLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(30, 160, 640, 480))
        self.imageLabel.setText("")
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setObjectName("imageLabel")
        self.pushButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 690, 72, 68))
        self.pushButton.setStyleSheet("")
        self.pushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("angle-left-solid.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QtCore.QSize(60, 60))
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(640, 670, 96, 41))
        font = QtGui.QFont()
        font.setFamily("Montserrat ExtraBold")
        font.setPointSize(25)
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setAutoFillBackground(False)
        self.label_4.setStyleSheet("color:rgb(255, 255, 255)")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(670, 710, 41, 41))
        font = QtGui.QFont()
        font.setFamily("Montserrat ExtraBold")
        font.setPointSize(25)
        font.setBold(True)
        self.label_5.setFont(font)
        self.label_5.setAutoFillBackground(False)
        self.label_5.setStyleSheet("color:rgb(226,33, 28)")
        self.label_5.setObjectName("label_5")
        self.imageLabel_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.imageLabel_2.setGeometry(QtCore.QRect(700, 160, 640, 480))
        self.imageLabel_2.setText("")
        self.imageLabel_2.setScaledContents(True)
        self.imageLabel_2.setObjectName("imageLabel_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1380, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Pump-"))
        self.label_2.setText(_translate("MainWindow", "it"))
        self.label_3.setText(_translate("MainWindow", "Squat"))
        self.label_4.setText(_translate("MainWindow", "Reps:"))
        self.label_5.setText(_translate("MainWindow", "0"))