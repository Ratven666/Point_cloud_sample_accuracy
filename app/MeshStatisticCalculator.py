from os import remove

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

    def save_statistic(self):
        statistic = self.df.describe()
        total_area = self.df["area"].sum()
        statistic.loc["TOTAL_MESH"] = [total_area, self.mesh.r, self.mesh.mse]
        statistic.to_csv(f"{self.mesh.mesh_name}_statistics.csv")
        return {"Total_area": total_area,
                "Count_of_r": self.mesh.r,
                "Cloud_MSE": self.mesh.mse,
                "Min_MSE": statistic["rmse"]["min"],
                "Max_MSE": statistic["rmse"]["max"],
                "Median_MSE": statistic["rmse"]["50%"]}

    def save_area_distributions_histograms(self):
        sns_plot = sns.displot(self.df, x="area", kde=True)
        sns_plot.savefig(f"{self.mesh.mesh_name}_area_distribution.png")

    def save_r_distributions_histograms(self):
        sns_plot = sns.displot(self.df, x="r", kde=True)
        sns_plot.savefig(f"{self.mesh.mesh_name}_r_distribution.png")

    def save_rmse_distributions_histograms(self):
        sns_plot = sns.displot(self.df, x="rmse", kde=True)
        sns_plot.savefig(f"{self.mesh.mesh_name}_rmse_distribution.png")

    def save_pair_plot_distributions_histograms(self):
        pair_grid = sns.PairGrid(self.df)
        pair_grid.map_upper(sns.histplot)
        pair_grid.map_lower(sns.kdeplot, fill=True)
        pair_grid.map_diag(sns.histplot, kde=True)
        pair_grid.savefig(f"{self.mesh.mesh_name}_pair_grid.png")
