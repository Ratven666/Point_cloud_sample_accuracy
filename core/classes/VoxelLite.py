from core.classes.ScanLite import ScanLite
from core.classes.abc_classes.VoxelABC import VoxelABC


class VoxelLite(VoxelABC):
    """
    Воксель не связанный с базой данных
    """
    __slots__ = ["id", "X", "Y", "Z", "step", "vxl_mdl_id", "vxl_name", "scan_id", "len", "R", "G", "B"]

    def __init__(self, X, Y, Z, step, vxl_mdl_id):
        super().__init__(X, Y, Z, step, vxl_mdl_id)
        self.scan = ScanLite(f"SC_{self.vxl_name}")

    @classmethod
    def parse_voxel_from_data_row(cls, row):
        voxel = cls(X=row[1], Y=row[2], Z=row[3], step=row[4], vxl_mdl_id=row[5])
        voxel.id = row[0]
        voxel.vxl_name = row[6]
        voxel.scan_id = row[7]
        voxel.len = row[8]
        voxel.R = row[9]
        voxel.G = row[10]
        voxel.B = row[11]
        return voxel

