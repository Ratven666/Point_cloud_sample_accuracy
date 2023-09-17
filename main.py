from os import path
import warnings

from app.CsvMeshDataExporter import CsvMeshDataExporter
from app.MeshStatisticCalculator import MeshStatisticCalculator
from app.core.classes.MeshLite import MeshLite
from app.core.classes.MeshSegmentModelDB import MeshSegmentModelDB
from app.core.classes.ScanDB import ScanDB
from app.core.classes.VoxelModelDB import VoxelModelDB
from app.core.utils.mesh_utils.mesh_exporters.PlyMseMeshExporter import PlyMseMeshExporter
from app.core.utils.start_db import create_db


def main():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    create_db()
    # base_file_path = "src/KuchaRGB_0_10.txt"
    base_file_path = "src/KuchaRGB_0_05.txt"
    sample_file_path = "src/KuchaRGB_05.txt"

    base_file_name = path.basename(base_file_path).split(".")[0]
    sample_file_name = path.basename(sample_file_path).split(".")[0]

    base_scan = ScanDB(f"Base_scan_{base_file_name}")
    base_scan.load_scan_from_file(file_name=base_file_path)

    sampled_scan = ScanDB(f"{sample_file_name}")
    sampled_scan.load_scan_from_file(file_name=sample_file_path)

    vm = VoxelModelDB(base_scan, 0.25, dx=0, dy=0, dz=0, is_2d_vxl_mdl=True)

    mesh = MeshLite(sampled_scan)
    print(mesh)
    # print(mesh)
    mesh_sm = MeshSegmentModelDB(vm, mesh)
    print(mesh_sm)

    mesh.calk_mesh_mse(mesh_sm)
    print(mesh)

    PlyMseMeshExporter(mesh, min_mse=0.01, max_mse=0.05).export()

    CsvMeshDataExporter(mesh).export_mesh_data()
    MeshStatisticCalculator(mesh).save_distributions_histograms()
    MeshStatisticCalculator(mesh).save_statistic()


if __name__ == "__main__":
    main()
