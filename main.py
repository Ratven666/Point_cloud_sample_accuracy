from core.classes.MeshLite import MeshLite
from core.classes.MeshSegmentModelDB import MeshSegmentModelDB
from core.classes.ScanDB import ScanDB
from core.classes.VoxelModelDB import VoxelModelDB
from core.utils.mesh_utils.mesh_exporters.PlyMseMeshExporter import PlyMseMeshExporter
from core.utils.start_db import create_db


def main():
    create_db()

    scan_for_mesh = ScanDB("KuchaRGB_05_1")
    scan_for_mesh.load_scan_from_file(file_name="src/KuchaRGB_05.txt")

    scan = ScanDB("KuchaRGB")
    scan.load_scan_from_file(file_name="src/KuchaRGB_0_10.txt")

    vm = VoxelModelDB(scan, 0.25, dx=0, dy=0, dz=0, is_2d_vxl_mdl=True)

    #
    mesh = MeshLite(scan_for_mesh)
    print(mesh)
    # print(mesh)
    mesh_sm = MeshSegmentModelDB(vm, mesh)
    print(mesh_sm)

    mesh.calk_mesh_mse(mesh_sm)
    print(mesh)
    #
    PlyMseMeshExporter(mesh, min_mse=0.01, max_mse=0.05).export()


if __name__ == "__main__":
    main()
