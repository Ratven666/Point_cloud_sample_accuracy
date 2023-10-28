
from os import path, remove
from pathlib import Path

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QFileDialog, QTableWidgetItem, QMessageBox

from app.CsvMeshDataExporter import CsvMeshDataExporter
from app.MeshStatisticCalculator import MeshStatisticCalculator
from app.core.CONFIG import DATABASE_NAME
from app.core.classes.MeshLite import MeshLite
from app.core.classes.MeshSegmentModelDB import MeshSegmentModelDB
from app.core.classes.ScanDB import ScanDB
from app.core.classes.VoxelModelDB import VoxelModelDB
from app.core.utils.mesh_utils.mesh_exporters.PlyMseMeshExporter import PlyMseMeshExporter
from app.core.utils.mesh_utils.mesh_filters.MaxEdgeLengthMeshFilter import MaxEdgeLengthMeshFilter
from app.core.utils.start_db import create_db, engine


class UiPointCloudAnalizer(QWidget):

    def __init__(self):
        super().__init__()
        self.setupUi()
        self.base_filepath = None
        self.sample_filepath = None
        self.max_border_length_m = self.slider_max_edge_filter.value()
        self.base_file_path_button.clicked.connect(self.open_file_dialog_base_filepath)
        self.base_file_path_text.textChanged.connect(self.base_filepath_from_text_line)
        self.sample_file_path_button.clicked.connect(self.open_file_dialog_sample_filepath)
        self.sample_file_path_text.textChanged.connect(self.sample_filepath_from_text_line)
        self.cb_main_graf_distr.toggled.connect(self.cb_mse_graf.setEnabled)
        self.cb_main_graf_distr.toggled.connect(self.cb_r_graf.setEnabled)
        self.cb_main_graf_distr.toggled.connect(self.cb_area_graf.setEnabled)
        self.cb_main_graf_distr.toggled.connect(self.cb_pair_plot_graf.setEnabled)
        self.cb_main_graf_distr.toggled.connect(lambda checked: not checked and
                                                                self.cb_mse_graf.setChecked(False))
        self.cb_main_graf_distr.toggled.connect(lambda checked: not checked and
                                                                self.cb_r_graf.setChecked(False))
        self.cb_main_graf_distr.toggled.connect(lambda checked: not checked and
                                                                self.cb_area_graf.setChecked(False))
        self.cb_main_graf_distr.toggled.connect(lambda checked: not checked and
                                                                self.cb_pair_plot_graf.setChecked(False))
        self.cb_maxedge_filter.toggled.connect(self.slider_max_edge_filter.setEnabled)
        self.cb_maxedge_filter.toggled.connect(self.sb_max_edge_filter.setEnabled)
        self.cb_maxedge_filter.toggled.connect(self.label_9.setEnabled)
        self.cb_maxedge_filter.toggled.connect(self.label_11.setEnabled)
        self.slider_max_edge_filter.valueChanged.connect(self.sliders_update)
        self.sb_max_edge_filter.valueChanged.connect(self.sb_update)




        self.cb_main_tin_surface.toggled.connect(self.sb_min_mse_tin_surf.setEnabled)
        self.cb_main_tin_surface.toggled.connect(self.label_min_mse_tin_surf.setEnabled)
        self.cb_main_tin_surface.toggled.connect(self.sb_max_mse_tin_surf.setEnabled)
        self.cb_main_tin_surface.toggled.connect(self.label_max_mse_tin_surf.setEnabled)
        self.cb_main_tin_surface.toggled.connect(lambda checked: not checked and
                                                                self.sb_min_mse_tin_surf.setValue(0.00))
        self.cb_main_tin_surface.toggled.connect(lambda checked: not checked and
                                                                self.sb_max_mse_tin_surf.setValue(99.99))
        self.progress = 0
        self.start_button.clicked.connect(self.start_calculation)

    def start_calculation(self):
        self.start_button.setEnabled(False)
        self.result_table.setEnabled(False)
        self.progress = 0
        self.progressBar.setProperty("value", 0)
        create_db()
        base_file_name = path.basename(self.base_filepath).split(".")[0]
        sample_file_name = path.basename(self.sample_filepath).split(".")[0]
        dir_path = path.dirname(self.sample_filepath)
        self.progressBar.setProperty("value", 1)
        base_scan = ScanDB(f"Base_scan_{base_file_name}")
        base_scan.load_scan_from_file(file_name=self.base_filepath)
        self.progressBar.setProperty("value", 15)
        sampled_scan = ScanDB(sample_file_name)
        sampled_scan.load_scan_from_file(file_name=self.sample_filepath)
        self.progressBar.setProperty("value", 20)
        vm = VoxelModelDB(base_scan,
                          self.__calk_voxel_size(base_scan, sampled_scan),
                          dx=0, dy=0, dz=0, is_2d_vxl_mdl=True)
        self.progressBar.setProperty("value", 40)
        mesh = MeshLite(sampled_scan)
        if self.cb_maxedge_filter.isChecked():
            MaxEdgeLengthMeshFilter(mesh, self.max_border_length_m).filter_mesh()
        self.progressBar.setProperty("value", 50)
        mesh_sm = MeshSegmentModelDB(vm, mesh)
        self.progressBar.setProperty("value", 70)
        mesh.calk_mesh_mse(mesh_sm)
        self.progressBar.setProperty("value", 80)
        ####################################################
        csv_exp = CsvMeshDataExporter(mesh)
        csv_exp.export_mesh_data(file_path=dir_path)
        stat_calculator = MeshStatisticCalculator(mesh)
        stat_dict = stat_calculator.get_statistic()
        stat_calculator.save_statistic(file_path=dir_path)
        self.progressBar.setProperty("value", 85)
        ####################################################
        graf_dict = {"rmse": self.cb_mse_graf.isChecked(),
                     "r": self.cb_r_graf.isChecked(),
                     "area": self.cb_area_graf.isChecked(),
                     "pair_plot": self.cb_pair_plot_graf.isChecked()}
        stat_calculator.save_distributions_histograms(graf_dict, file_path=dir_path)
        self.progressBar.setProperty("value", 95)
        ####################################################
        if self.cb_main_tin_surface.isChecked():
            min_mse = None if self.sb_min_mse_tin_surf.value() == 0 else self.sb_min_mse_tin_surf.value()
            max_mse = None if self.sb_max_mse_tin_surf.value() == 99.99 else self.sb_max_mse_tin_surf.value()
            PlyMseMeshExporter(mesh, min_mse=min_mse, max_mse=max_mse).export(file_path=dir_path)
        self.progressBar.setProperty("value", 100)
        ###################################################
        if self.cb_save_full_tin_csv_log.isChecked() is False:
            remove(path.join(dir_path, csv_exp.file_name))
        if self.cb_save_db.isChecked() is False:
            engine.dispose()
            remove(path.join(".", DATABASE_NAME))
        ###################################################
        self.result_table.setEnabled(True)
        try:
            self.result_table.setItem(0, 0, QTableWidgetItem(str(round(stat_dict["Total_area"], 4))))
            self.result_table.setItem(0, 1, QTableWidgetItem(str(stat_dict["Count_of_r"])))
            self.result_table.setItem(0, 2, QTableWidgetItem(str(round(stat_dict["Cloud_MSE"], 4))))
            self.result_table.setItem(0, 3, QTableWidgetItem(str(round(stat_dict["Min_MSE"], 4))))
            self.result_table.setItem(0, 4, QTableWidgetItem(str(round(stat_dict["Max_MSE"], 4))))
            self.result_table.setItem(0, 5, QTableWidgetItem(str(round(stat_dict["Median_MSE"], 4))))
        except TypeError:
            self.result_table.setItem(0, 0, QTableWidgetItem(str(None)))
            self.result_table.setItem(0, 1, QTableWidgetItem(str(None)))
            self.result_table.setItem(0, 2, QTableWidgetItem(str(None)))
            self.result_table.setItem(0, 3, QTableWidgetItem(str(None)))
            self.result_table.setItem(0, 4, QTableWidgetItem(str(None)))
            self.result_table.setItem(0, 5, QTableWidgetItem(str(None)))
        ###################################################
        self.start_button.setEnabled(True)
        dig = QMessageBox(self)
        dig.setWindowTitle("Result")
        dig.setText("Расчет завершен!")
        dig.setIcon(QMessageBox.Icon.Information)
        dig.exec()

    @staticmethod
    def __calk_voxel_size(base_scan, sampled_scan):
        area = (base_scan.max_X - base_scan.min_X) * (base_scan.max_Y - base_scan.min_Y)
        voxel_size = area / sampled_scan.len
        voxel_size = round((voxel_size // 0.05 + 1) * 0.05, 2)
        return voxel_size


    def sb_update(self):
        self.max_border_length_m = self.sb_max_edge_filter.value()
        self.slider_max_edge_filter.setValue(int(self.max_border_length_m))

    def sliders_update(self):
        self.max_border_length_m = self.slider_max_edge_filter.value()
        self.sb_max_edge_filter.setValue(self.max_border_length_m)

    def base_filepath_from_text_line(self):
        self.base_filepath = self.base_file_path_text.toPlainText()
        if self.base_filepath and self.sample_filepath:
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)

    def sample_filepath_from_text_line(self):
        self.sample_filepath = self.sample_file_path_text.toPlainText()
        if self.base_filepath and self.sample_filepath:
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)

    def open_file_dialog_base_filepath(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            ".",
            "PointCloud (*.txt *.ascii)"
        )
        if filename:
            path = Path(filename)
            self.base_file_path_text.setText(str(filename))
            self.base_filepath = str(path)

    def open_file_dialog_sample_filepath(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            ".",
            "PointCloud (*.txt *.ascii)"
        )
        if filename:
            path = Path(filename)
            self.sample_file_path_text.setText(str(filename))
            self.sample_filepath = str(path)

    def setupUi(self):
        self.setWindowIcon(QIcon("icon.ico"))
        self.setObjectName("PointCloudAnalizer")
        self.setEnabled(True)
        self.resize(925, 600)
        self.setMinimumSize(QtCore.QSize(925, 0))
        self.setMaximumSize(QtCore.QSize(925, 600))
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.base_file_path_button = QtWidgets.QToolButton(parent=self)
        self.base_file_path_button.setObjectName("base_file_path_button")
        self.gridLayout_4.addWidget(self.base_file_path_button, 0, 2, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.line = QtWidgets.QFrame(parent=self)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 1, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=self)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(parent=self)
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 16))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.cb_main_graf_distr = QtWidgets.QCheckBox(parent=self)
        self.cb_main_graf_distr.setMaximumSize(QtCore.QSize(16777215, 16))
        self.cb_main_graf_distr.setObjectName("cb_main_graf_distr")
        self.gridLayout_3.addWidget(self.cb_main_graf_distr, 1, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem2, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem3, 0, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_3, 0, 2, 1, 1)
        self.line_3 = QtWidgets.QFrame(parent=self)
        self.line_3.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_2.addWidget(self.line_3, 1, 3, 1, 1)
        self.cb_r_graf = QtWidgets.QCheckBox(parent=self)
        self.cb_r_graf.setEnabled(False)
        self.cb_r_graf.setObjectName("cb_r_graf")
        self.gridLayout_2.addWidget(self.cb_r_graf, 3, 2, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_max_mse_tin_surf = QtWidgets.QLabel(parent=self)
        self.label_max_mse_tin_surf.setEnabled(False)
        self.label_max_mse_tin_surf.setObjectName("label_max_mse_tin_surf")
        self.horizontalLayout_4.addWidget(self.label_max_mse_tin_surf)
        self.sb_max_mse_tin_surf = QtWidgets.QDoubleSpinBox(parent=self)
        self.sb_max_mse_tin_surf.setEnabled(False)
        self.sb_max_mse_tin_surf.setMaximumSize(QtCore.QSize(48, 16777215))
        self.sb_max_mse_tin_surf.setSingleStep(0.1)
        self.sb_max_mse_tin_surf.setProperty("value", 99.99)
        self.sb_max_mse_tin_surf.setObjectName("sb_max_mse_tin_surf")
        self.horizontalLayout_4.addWidget(self.sb_max_mse_tin_surf)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 3, 3, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_min_mse_tin_surf = QtWidgets.QLabel(parent=self)
        self.label_min_mse_tin_surf.setEnabled(False)
        self.label_min_mse_tin_surf.setObjectName("label_min_mse_tin_surf")
        self.horizontalLayout.addWidget(self.label_min_mse_tin_surf)
        self.sb_min_mse_tin_surf = QtWidgets.QDoubleSpinBox(parent=self)
        self.sb_min_mse_tin_surf.setEnabled(False)
        self.sb_min_mse_tin_surf.setMaximumSize(QtCore.QSize(48, 16777215))
        self.sb_min_mse_tin_surf.setSingleStep(0.1)
        self.sb_min_mse_tin_surf.setObjectName("sb_min_mse_tin_surf")
        self.horizontalLayout.addWidget(self.sb_min_mse_tin_surf)
        self.gridLayout_2.addLayout(self.horizontalLayout, 2, 3, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.cb_main_tin_surface = QtWidgets.QCheckBox(parent=self)
        self.cb_main_tin_surface.setObjectName("cb_main_tin_surface")
        self.gridLayout_5.addWidget(self.cb_main_tin_surface, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=self)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 0, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_5.addItem(spacerItem4, 0, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_5.addItem(spacerItem5, 0, 2, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout_5)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 3, 1, 1)
        self.line_2 = QtWidgets.QFrame(parent=self)
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_2.addWidget(self.line_2, 1, 2, 1, 1)
        self.cb_area_graf = QtWidgets.QCheckBox(parent=self)
        self.cb_area_graf.setEnabled(False)
        self.cb_area_graf.setObjectName("cb_area_graf")
        self.gridLayout_2.addWidget(self.cb_area_graf, 4, 2, 1, 1)
        self.cb_save_db = QtWidgets.QCheckBox(parent=self)
        self.cb_save_db.setEnabled(True)
        self.cb_save_db.setObjectName("cb_save_db")
        self.gridLayout_2.addWidget(self.cb_save_db, 3, 0, 1, 1)
        self.cb_mse_graf = QtWidgets.QCheckBox(parent=self)
        self.cb_mse_graf.setEnabled(False)
        self.cb_mse_graf.setObjectName("cb_mse_graf")
        self.gridLayout_2.addWidget(self.cb_mse_graf, 2, 2, 1, 1)
        self.cb_save_full_tin_csv_log = QtWidgets.QCheckBox(parent=self)
        self.cb_save_full_tin_csv_log.setEnabled(True)
        self.cb_save_full_tin_csv_log.setObjectName("cb_save_full_tin_csv_log")
        self.gridLayout_2.addWidget(self.cb_save_full_tin_csv_log, 2, 0, 1, 1)
        self.cb_pair_plot_graf = QtWidgets.QCheckBox(parent=self)
        self.cb_pair_plot_graf.setEnabled(False)
        self.cb_pair_plot_graf.setObjectName("cb_pair_plot_graf")
        self.gridLayout_2.addWidget(self.cb_pair_plot_graf, 5, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 3, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(parent=self)
        self.label_10.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                   QtCore.Qt.AlignmentFlag.AlignTrailing |
                                   QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 1, 0, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        spacerItem6 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_5.addItem(spacerItem6)
        self.base_file_path_text = QtWidgets.QTextEdit(parent=self)
        self.base_file_path_text.setMaximumSize(QtCore.QSize(16777215, 25))
        self.base_file_path_text.setObjectName("base_file_path_text")
        self.verticalLayout_5.addWidget(self.base_file_path_text)
        spacerItem7 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_5.addItem(spacerItem7)
        self.gridLayout_4.addLayout(self.verticalLayout_5, 0, 1, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_4.addLayout(self.verticalLayout_2, 3, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(parent=self)
        self.label_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                  QtCore.Qt.AlignmentFlag.AlignTrailing |
                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_4.addWidget(self.label_6, 3, 0, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.sample_file_path_button = QtWidgets.QToolButton(parent=self)
        self.sample_file_path_button.setObjectName("sample_file_path_button")
        self.verticalLayout_3.addWidget(self.sample_file_path_button)
        self.gridLayout_4.addLayout(self.verticalLayout_3, 1, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(parent=self)
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 100))
        self.label_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                  QtCore.Qt.AlignmentFlag.AlignTrailing |
                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_6.addItem(spacerItem8)
        self.sample_file_path_text = QtWidgets.QTextEdit(parent=self)
        self.sample_file_path_text.setMaximumSize(QtCore.QSize(16777215, 25))
        self.sample_file_path_text.setObjectName("sample_file_path_text")
        self.verticalLayout_6.addWidget(self.sample_file_path_text)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_6.addItem(spacerItem9)
        self.gridLayout_4.addLayout(self.verticalLayout_6, 1, 1, 1, 1)
        self.line_4 = QtWidgets.QFrame(parent=self)
        self.line_4.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout_4.addWidget(self.line_4, 2, 1, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout_4)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_12 = QtWidgets.QLabel(parent=self)
        self.label_12.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                   QtCore.Qt.AlignmentFlag.AlignTrailing |
                                   QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_7.addWidget(self.label_12)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_4 = QtWidgets.QLabel(parent=self)
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                  QtCore.Qt.AlignmentFlag.AlignTrailing |
                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_8.addWidget(self.label_4)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_5 = QtWidgets.QLabel(parent=self)
        self.label_5.setObjectName("label_5")
        self.gridLayout_7.addWidget(self.label_5, 0, 1, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                             QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_7.addItem(spacerItem10, 0, 0, 1, 1)
        self.cb_maxedge_filter = QtWidgets.QCheckBox(parent=self)
        self.cb_maxedge_filter.setObjectName("cb_maxedge_filter")
        self.gridLayout_7.addWidget(self.cb_maxedge_filter, 1, 1, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                             QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_7.addItem(spacerItem11, 0, 3, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_7, 1, 1, 1, 1)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_11 = QtWidgets.QLabel(parent=self)
        self.label_11.setEnabled(False)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_6.addWidget(self.label_11)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                             QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem12)
        self.label_9 = QtWidgets.QLabel(parent=self)
        self.label_9.setEnabled(False)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_6.addWidget(self.label_9)
        self.gridLayout_8.addLayout(self.horizontalLayout_6, 2, 0, 1, 1)
        self.slider_max_edge_filter = QtWidgets.QSlider(parent=self)
        self.slider_max_edge_filter.setEnabled(False)
        self.slider_max_edge_filter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_max_edge_filter.setObjectName("slider_max_edge_filter")
        self.gridLayout_8.addWidget(self.slider_max_edge_filter, 4, 0, 1, 1)
        self.sb_max_edge_filter = QtWidgets.QDoubleSpinBox(parent=self)
        self.sb_max_edge_filter.setEnabled(False)
        self.sb_max_edge_filter.setObjectName("sb_max_edge_filter")
        self.gridLayout_8.addWidget(self.sb_max_edge_filter, 4, 1, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_8, 2, 1, 1, 1)
        self.horizontalLayout_8.addLayout(self.gridLayout_6)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_7.addLayout(self.verticalLayout)
        self.verticalLayout_4.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.progressBar = QtWidgets.QProgressBar(parent=self)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_3.addWidget(self.progressBar)
        self.start_button = QtWidgets.QPushButton(parent=self)
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("background-color: rgb(170, 255, 127);")
        self.start_button.setFlat(False)
        self.start_button.setObjectName("start_button")
        self.horizontalLayout_3.addWidget(self.start_button)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.result_table = QtWidgets.QTableWidget(parent=self)
        self.result_table.setEnabled(False)
        self.result_table.setObjectName("result_table")
        self.result_table.setColumnCount(6)
        self.result_table.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.result_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(5, item)
        self.verticalLayout_4.addWidget(self.result_table)
        self.horizontalLayout_5.addLayout(self.verticalLayout_4)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, PointCloudAnalizer):
        _translate = QtCore.QCoreApplication.translate
        PointCloudAnalizer.setWindowTitle(_translate("PointCloudAnalizer", "Form"))
        self.base_file_path_button.setText(_translate("PointCloudAnalizer", "..."))
        self.label_2.setText(_translate("PointCloudAnalizer", "Сохраняемая информация"))
        self.cb_main_graf_distr.setText(_translate("PointCloudAnalizer", "Построить графики распределения"))
        self.label.setText(_translate("PointCloudAnalizer", "Графики распределения параметров TIN поверхности"))
        self.cb_r_graf.setText(_translate("PointCloudAnalizer", "Распределение количества\n"
"избыточных данных"))
        self.label_max_mse_tin_surf.setText(_translate("PointCloudAnalizer", "Максимальное значение шкалы СКП"))
        self.label_min_mse_tin_surf.setText(_translate("PointCloudAnalizer", "Минимальное значение шкалы СКП"))
        self.cb_main_tin_surface.setText(_translate("PointCloudAnalizer", "Создать"))
        self.label_3.setText(_translate("PointCloudAnalizer", "Создать TIN поверхность\n"
"распределения СКП "))
        self.cb_area_graf.setText(_translate("PointCloudAnalizer", "Распределение площади\n"
"треугольников в поверхности"))
        self.cb_save_db.setText(_translate("PointCloudAnalizer", "Сохранить служебную базу ланных"))
        self.cb_mse_graf.setText(_translate("PointCloudAnalizer", "Распределение СКП\n"
"в треугольниках поверхности"))
        self.cb_save_full_tin_csv_log.setText(_translate("PointCloudAnalizer", "Сохранить полное описание анализируемой\n"
"поверхностив CSV файле"))
        self.cb_pair_plot_graf.setText(_translate("PointCloudAnalizer", "Совмещенный график\n"
"распределения"))
        self.label_10.setText(_translate("PointCloudAnalizer", "Разреженное\n"
"облако:"))
        self.label_6.setText(_translate("PointCloudAnalizer", "Настройки\n"
"выводимой\n"
"информации"))
        self.sample_file_path_button.setText(_translate("PointCloudAnalizer", "..."))
        self.label_7.setText(_translate("PointCloudAnalizer", "Исходное\n"
"облако:"))
        self.label_12.setText(_translate("PointCloudAnalizer", "Настройки\n"
"TIN\n"
"поверхности"))
        self.label_4.setText(_translate("PointCloudAnalizer", "Макс.\n"
"сторона\n"
"полигона, м:"))
        self.label_5.setText(_translate("PointCloudAnalizer", "Включить фильтрацию по максимальной "
                                                              "длинне ребра TIN поверхности"))
        self.cb_maxedge_filter.setText(_translate("PointCloudAnalizer", "Включить"))
        self.label_11.setText(_translate("PointCloudAnalizer", "Меньше"))
        self.label_9.setText(_translate("PointCloudAnalizer", "Больше"))
        self.start_button.setText(_translate("PointCloudAnalizer", "Запуск анализа"))
        item = self.result_table.verticalHeaderItem(0)
        item.setText(_translate("PointCloudAnalizer", "Рассчитанные значения"))
        item = self.result_table.horizontalHeaderItem(0)
        item.setText(_translate("PointCloudAnalizer", "Общая площадь, м2"))
        item = self.result_table.horizontalHeaderItem(1)
        item.setText(_translate("PointCloudAnalizer", "Кол-во изб. данных"))
        item = self.result_table.horizontalHeaderItem(2)
        item.setText(_translate("PointCloudAnalizer", "СКП облака, м"))
        item = self.result_table.horizontalHeaderItem(3)
        item.setText(_translate("PointCloudAnalizer", "Мин. СКП, м"))
        item = self.result_table.horizontalHeaderItem(4)
        item.setText(_translate("PointCloudAnalizer", "Макс. СКП, м"))
        item = self.result_table.horizontalHeaderItem(5)
        item.setText(_translate("PointCloudAnalizer", "Медиана СКП, м"))
