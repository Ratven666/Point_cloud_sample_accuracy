from os import remove, path

import pandas as pd
import seaborn as sns

from app.CsvMeshDataExporter import CsvMeshDataExporter


class MeshStatisticCalculator:

    def __init__(self, mesh):
        sns.set_style("darkgrid")
        self.mesh = mesh
        try:
            self.df = pd.read_csv(f"{self.mesh.mesh_name}.csv", delimiter=",")[["area", "r", "rmse"]]
        except FileNotFoundError:
            file_name = CsvMeshDataExporter(mesh).export_mesh_data()
            self.df = pd.read_csv(file_name, delimiter=",")[["area", "r", "rmse"]]
            remove(file_name)

    def get_statistic(self):
        statistic = self.df.describe()
        total_area = self.df["area"].sum()
        statistic.loc["TOTAL_MESH"] = [total_area, self.mesh.r, self.mesh.mse]
        try:
            return {"Total_area": total_area,
                    "Count_of_r": self.mesh.r,
                    "Cloud_MSE": self.mesh.mse,
                    "Min_MSE": statistic["rmse"]["min"],
                    "Max_MSE": statistic["rmse"]["max"],
                    "Median_MSE": statistic["rmse"]["50%"]}
        except KeyError:
            return None

    def save_statistic(self, file_path="."):
        statistic = self.df.describe()
        total_area = self.df["area"].sum()
        statistic.loc["TOTAL_MESH"] = [total_area, self.mesh.r, self.mesh.mse]
        file_path = path.join(file_path, f"{self.mesh.mesh_name}_statistics.csv")
        statistic.to_csv(file_path)

    def save_area_distributions_histograms(self, file_path="."):
        sns_plot = sns.displot(self.df, x="area", kde=True)
        file_path = path.join(file_path, f"{self.mesh.mesh_name}_area_distribution.png")
        sns_plot.savefig(file_path)

    def save_r_distributions_histograms(self, file_path="."):
        sns_plot = sns.displot(self.df, x="r", kde=True)
        file_path = path.join(file_path, f"{self.mesh.mesh_name}_r_distribution.png")
        sns_plot.savefig(file_path)

    def save_rmse_distributions_histograms(self, file_path="."):
        sns_plot = sns.displot(self.df, x="rmse", kde=True)
        file_path = path.join(file_path, f"{self.mesh.mesh_name}_rmse_distribution.png")
        sns_plot.savefig(file_path)

    def save_pair_plot_distributions_histograms(self, file_path="."):
        pair_grid = sns.PairGrid(self.df)
        pair_grid.map_upper(sns.histplot)
        pair_grid.map_lower(sns.kdeplot, fill=True)
        pair_grid.map_diag(sns.histplot, kde=True)
        file_path = path.join(file_path, f"{self.mesh.mesh_name}_pair_grid.png")
        pair_grid.savefig(file_path)

    def save_distributions_histograms(self, graf_dict, file_path="."):
        if graf_dict["rmse"]:
            self.save_rmse_distributions_histograms(file_path=file_path)
        if graf_dict["r"]:
            self.save_r_distributions_histograms(file_path=file_path)
        if graf_dict["area"]:
            self.save_area_distributions_histograms(file_path=file_path)
        if graf_dict["pair_plot"]:
            self.save_pair_plot_distributions_histograms(file_path=file_path)
