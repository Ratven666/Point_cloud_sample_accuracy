from os import path

from core.utils.mesh_utils.mesh_exporters.MeshExporterABC import MeshExporterABC


class PlyMeshExporter(MeshExporterABC):

    def __init__(self, mesh):
        super().__init__(mesh)

    def __create_header(self):
        return f"ply\n" \
               f"format ascii 1.0\n" \
               f"comment author: Mikhail Vystrchil\n" \
               f"comment object: {self.mesh.mesh_name}\n" \
               f"element vertex {len(self.vertices)}\n" \
               f"property float x\n" \
               f"property float y\n" \
               f"property float z\n" \
               f"property uchar red\n" \
               f"property uchar green\n" \
               f"property uchar blue\n" \
               f"element face {len(self.faces)}\n" \
               f"property list uchar int vertex_index\n" \
               f"end_header\n"

    def __create_vertices_str(self):
        vertices_str = ""
        for idx in range(len(self.vertices)):
            vertices_str += f"{self.vertices[idx][0]} {self.vertices[idx][1]} {self.vertices[idx][2]} " \
                            f"{self.vertices_colors[idx][0]} " \
                            f"{self.vertices_colors[idx][1]} " \
                            f"{self.vertices_colors[idx][2]}\n"
        return vertices_str

    def __create_faces_str(self):
        faces_str = ""
        for face in self.faces:
            faces_str += f"3 {face[0]} {face[1]} {face[2]}\n"
        return faces_str

    def _save_ply(self, file_path):
        with open(file_path, "wb") as file:
            file.write(self.__create_header().encode("ascii"))
            file.write(self.__create_vertices_str().encode("ascii"))
            file.write(self.__create_faces_str().encode("ascii"))

    def export(self, file_path="."):
        file_path = path.join(file_path, f"{self.mesh.mesh_name.replace(':', '=')}.ply")
        self._save_ply(file_path)
