import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap
import plotly.io as pio
import os

class ContourPlot():
    """
    ContourPlot class

    This class generates and saves contour plots based on a CSV file containing 2D data. The CSV file should include a column for the y-axis (typically representing chemical shifts) and multiple columns for the x-axis (typically representing time steps). The class uses the data from the CSV file to create a contour plot and supports customization of the color mapping and plot dimensions.

    Attributes:
        file_path (str): Path to the CSV file containing the data to be plotted.
        df (pandas.DataFrame): The dataframe containing the data, loaded from the CSV file.
        basename (str): The base name of the file (extracted from `file_path`).
        plot_dir (Path): The directory where the plot will be saved, located under 'output/{basename}_output/plots'.
        contour_pdf (Path): The path for saving the generated contour plot as a PDF file.
        Z (numpy.ndarray): The 2D array of values to be plotted, excluding the first column (time steps).
        X (numpy.ndarray): The 2D array of x-axis values (time steps), generated from the columns of `Z`.
        Y (numpy.ndarray): The 2D array of y-axis values (chemical shifts), generated from the first column of `df`.

    Methods:
        plot(zmin, zmax):
            Generates a contour plot with customized colormap and axis labels. The intensity is scaled based on `zmin` and `zmax`.
            
            Parameters:
                zmin (float): Minimum scaling factor for the intensity.
                zmax (float): Maximum scaling factor for the intensity.
            
            Returns:
                matplotlib.figure.Figure: The generated contour plot figure.

        save_fig(fig, name, width=1200, height=800):
            Saves the given figure as both a PDF and PNG file.
            
            Parameters:
                fig (matplotlib.figure.Figure): The figure to save.
                name (str): The name to use for the saved file (without extension).
                width (int, optional): The width of the saved figure in pixels (default is 1200).
                height (int, optional): The height of the saved figure in pixels (default is 800).
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.basename = os.path.basename(self.file_path)
        self.plot_dir = Path('output', self.basename + '_output', 'plots')
        self.contour_pdf = Path(self.plot_dir, f'Contour_{self.basename}')

        # Assuming `self.df` is a DataFrame and `self.Z` is created from its data (excluding the first column)
        self.Z = self.df.iloc[:, 1:].to_numpy()

        # Generate x and y ranges
        x = np.arange(self.Z.shape[1])  # Columns correspond to x
        y = self.df.iloc[:,0]  # Rows correspond to y
        
        # Create the meshgrid
        self.X, self.Y = np.meshgrid(x, y)
           
    def plot(self, zmin, zmax):
        """
        plot(self, zmin, zmax)

        Generates a contour plot using the data from the object, applying a custom colormap and intensity scaling based on the specified `zmin` and `zmax` values. The plot is displayed with appropriate axis labels, a colorbar, and a grid.

        Parameters:
            zmin (float): Minimum scaling factor for the intensity. The minimum intensity will be set to `zmin * max(Z)`.
            zmax (float): Maximum scaling factor for the intensity. The maximum intensity will be set to `zmax * max(Z)`.

        Returns:
            matplotlib.figure.Figure: The generated contour plot figure.
            
        Description:
            - The function creates a contour plot of the data (stored in `self.Z`), with the x-axis and y-axis defined by `self.X` and `self.Y`, respectively.
            - A custom colormap is created by modifying the 'magma' colormap, setting the lowest value to white.
            - The plot includes a colorbar labeled 'Intensity' and displays the time steps and chemical shifts on the x-axis and y-axis.
            - The figure background is set to white, and the plot title includes the basename of the file.
        """

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('white')  # Set the figure background to white
        
        # Create a custom colormap based on 'magma'
        cmap = plt.cm.magma
        new_colors = cmap(np.linspace(0, 1, cmap.N))  # Get the colors of 'magma'
        new_colors[0] = np.array([1, 1, 1, 1])       # Set the lowest value (0) to white
        custom_cmap = ListedColormap(new_colors)

        # Plot with the custom colormap
        contour = ax.contourf(
            self.X,
            self.Y,
            self.Z,
            levels=20,
            cmap=custom_cmap,
            vmin=zmin * self.Z.max(),
            vmax=zmax * self.Z.max()
        )

        # Add a colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Intensity')

        ax.set_xlabel('Time step')
        ax.set_ylabel('Chemical shift [ppm]')
        ax.set_title(f'Contour plot of File {self.basename}')
        ax.grid(True)

        return fig
    
    def save_fig(self, fig, name, width=1200, height=800):
        """
        save_fig(self, fig, name, width=1200, height=800)

        Saves the provided figure (`fig`) to both PDF and PNG formats with the specified file name (`name`). The saved figure is resized to the specified width and height in pixels.

        Parameters:
            fig (matplotlib.figure.Figure): The figure to save.
            name (str): The base name (without extension) for the output files.
            width (int, optional): The width of the saved figure in pixels (default is 1200).
            height (int, optional): The height of the saved figure in pixels (default is 800).

        Description:
            - The function resizes the figure to the specified width and height, with a resolution of 300 DPI.
            - It saves the figure in both PDF and PNG formats using the specified file name (`name`).
            - The saved files will have the extensions `.pdf` and `.png` respectively.
        """
        
        fig.set_size_inches(width / 100, height / 100) # Set Output size 
        fig.savefig(f'{name}.pdf', format='pdf') # Save as PDF
        fig.savefig(f'{name}.png', format='png') # Save as PNG

