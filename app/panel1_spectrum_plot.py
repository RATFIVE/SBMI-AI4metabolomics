import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from LoadData import *
#from Layout import StreamlitApp
import streamlit as st
import plotly.io as pio
import plotly.express as px
from plotly.colors import sequential
from plotly.subplots import make_subplots

class Panel1SpectrumPlot():
    """
    Panel1SpectrumPlot class for generating and saving spectrum-related plots.

    This class is responsible for loading spectral data from CSV files and generating various plots related to the spectrum, noise, and fitted Lorentzian curves. It includes methods for creating plots for raw data, differences (noise), and sum fits for each frame, as well as saving these plots as PDF and PNG files.

    Attributes:
        file_path (str): The path to the CSV file containing the spectral data.
        file_name (str): The base name of the input file.
        plot_dir (Path): The directory where the plot files will be saved.
        spectrum_pdf (Path): Path to save the spectrum plot as a PDF.
        noise_pdf (Path): Path to save the noise plot as a PDF.
        fitted_pdf (Path): Path to save the fitted plot as a PDF.
        colors (list): List of colors for plot lines.
        template (str): Template for the plot layout.
        data (pd.DataFrame): Raw spectral data loaded from the input file.
        sum_data (pd.DataFrame): Data for sum fits loaded from 'sum_fit.csv'.
        differences (pd.DataFrame): Noise data loaded from 'differences.csv'.
        individual_fits (list): List of individual fit data frames, sorted by file name.

    Methods:
        __init__(file_path):
            Initializes the class by loading the spectral, fit, and noise data, setting up directories, and preparing file paths.
        
        plot(frame):
            Generates a combined plot with raw data, sum fits, and noise for a given frame.
        
        plot_raw(frame):
            Generates a plot of raw spectral data for a given frame.
        
        plot_diff(frame):
            Generates a plot of the differences (noise) for a given frame.
        
        plot_sum_fit(frame):
            Generates a plot of the sum fits and individual Lorentzian curves for a given frame.
        
        save_fig(fig, name):
            Saves the generated figure as both PDF and PNG files with the specified name.

    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.plot_dir = Path('output', self.file_name + '_output', 'plots')
        self.spectrum_pdf = Path(self.plot_dir, f'Spectrum_{self.file_name}')
        self.noise_pdf = Path(self.plot_dir, f'Noise{self.file_name}')
        self.fitted_pdf = Path(self.plot_dir, f'Fitted_{self.file_name}')
        self.colors = px.colors.qualitative.Dark24
        self.template = 'plotly_white' 

        # Ensure the plot directory exists 
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.data = pd.read_csv(file_path, header = 0)
        self.sum_data = pd.read_csv(Path('output', f'{self.file_name}_output', 'sum_fit.csv'), header = 0)
        self.differences = pd.read_csv(Path('output', f'{self.file_name}_output', 'differences.csv'), header = 0)
        
        # Read in and sort files numerically by the number in their names
        ind_file_names = sorted(
            os.listdir(Path('output', f'{self.file_name}_output', 'substance_fits')),
            key=lambda x: int(x.split('_fit_')[1].split('.csv')[0])  # Extract number for sorting
        )

        # Load the CSV files in the sorted order
        self.individual_fits = [
            pd.read_csv(Path('output', f'{self.file_name}_output', 'substance_fits', f), header=0)
            for f in ind_file_names
        ]

    def plot(self, frame):
        """
    Generates a combined spectrum plot for a given frame.

    This method creates a single plot that combines the raw spectral data, the sum of fitted Lorentzian curves, and the differences (noise) for a specified frame. The plot includes customized axis labels, title, and layout settings. It uses data from the raw, sum fit, and difference plots, adjusting the y-axis and x-axis ranges for consistency across the plot.

    Args:
        frame (int): The frame index for which the plot will be generated. It corresponds to a specific column in the spectral, sum fit, and difference data.

    Returns:
        plotly.graph_objects.Figure: The resulting figure containing the combined spectrum plot with raw data, sum fits, and differences for the specified frame.
        
    Attributes Set:
        min_y (float): The minimum y-value across all data for consistent y-axis scaling.
        max_y (float): The maximum y-value across all data for consistent y-axis scaling.
        min_x (float): The minimum x-value for the x-axis based on the data.
        max_x (float): The maximum x-value for the x-axis based on the data.
    """

        # for consistent y axis
        self.min_y = self.data.iloc[:,1:].min().min() + self.data.iloc[:,1:].min().min()
        self.max_y = self.data.iloc[:,1:].max().max()
        # for constant x axis
        self.min_x = self.data.iloc[:, 0].min()
        self.max_x = self.data.iloc[:, 0].max()
        
        # Initialize the figs with corresponding frame
        fig1 = self.plot_raw(frame)
        fig2 = self.plot_sum_fit(frame)
        fig3 = self.plot_diff(frame)
        
        fig = make_subplots(rows=1, cols=1, subplot_titles=(" ", " ", " "))
    
        # Add traces from fig_raw to the first subplot
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Add traces from fig_diff to the second subplot
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Add traces from fig_sm_fit to the third subplot
        for trace in fig3.data:
            fig.add_trace(trace, row=1, col=1)

        # Configure the Layout
        fig.update_layout(
            title=dict(
                text=f'Spectra for File {self.file_name} for Frame {frame}',
                font=dict(size=24)  # Font size for the title
            ),
            xaxis_title='Chemical shift [ppm]',
            yaxis_title='Intensity',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(
                range=[self.min_y, self.max_y],
                titlefont=dict(size=18),          # Font size for y-axis title
                tickfont=dict(size=18)            # Font size for y-axis ticks
            ),
            xaxis=dict(
                range=[self.max_x, self.min_x],
                dtick=0.5,
                titlefont=dict(size=18),          # Font size for x-axis title
                tickfont=dict(size=18)            # Font size for x-axis ticks
            )
                          
        )

        return fig

    def plot_raw(self, frame):

        """
        Generates a raw spectrum plot for a given frame.

        This method creates a plot displaying the raw spectrum data for the specified frame. The x-axis represents the chemical shift (in ppm), and the y-axis represents the intensity. The plot includes customization for the axis labels, title, and layout, and ensures that the x-axis and y-axis ranges are consistent with the overall data.

        Args:
            frame (int): The frame index for which the raw spectrum plot will be generated. It corresponds to a specific column in the raw data.

        Returns:
            plotly.graph_objects.Figure: The figure containing the raw spectrum plot for the specified frame.

        Attributes Set:
            min_y (float): The minimum y-value across all data for consistent y-axis scaling.
            max_y (float): The maximum y-value across all data for consistent y-axis scaling.
            min_x (float): The minimum x-value for the x-axis based on the data.
            max_x (float): The maximum x-value for the x-axis based on the data.
        """

        # Add Data to Fig
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.iloc[:,0][::-1],       # To Change direction of x axis from low to high 
                                 y=self.data.iloc[:,frame][::-1],   # To Change direction of x axis from low to high 
                                 mode='lines', name='Raw',
                                 line=dict(color=self.colors[0])
                                 ))
        # Setting the layout
        fig.update_layout(
            title='Spectrum',
            xaxis_title='Chemical shift [ppm]',
            yaxis_title='Intensity',
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(
                range=[self.min_y, self.max_y],
                titlefont=dict(size=18),          # Font size for y-axis title
                tickfont=dict(size=18)
                ),
            xaxis=dict(
                range=[self.max_x, self.min_x], 
                dtick=0.5,
                titlefont=dict(size=18),          # Font size for y-axis title
                tickfont=dict(size=18)
                ),              
            template=self.template
            )    

        return fig
    
    def plot_diff(self, frame):

        """
        Generates a plot for the difference (or noise) spectrum for a given frame.

        This method creates a plot that shows the difference (or noise) spectrum for the specified frame. The x-axis represents the chemical shift (in ppm), and the y-axis represents the intensity. The plot is customized with labels, title, and axis ranges, with a focus on visualizing the noise spectrum.

        Args:
            frame (int): The frame index for which the difference spectrum plot will be generated. It corresponds to a specific column in the differences data.

        Returns:
            plotly.graph_objects.Figure: The figure containing the difference (noise) spectrum plot for the specified frame.

        Attributes Set:
            max_y (float): The maximum y-value across all data for consistent y-axis scaling.
            min_x (float): The minimum x-value for the x-axis based on the data.
            max_x (float): The maximum x-value for the x-axis based on the data.
        """

        # Add data to Fig
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.differences.iloc[:,0][::-1],        # To Change direction of x axis from low to high 
                                 y=self.differences.iloc[:,frame][::-1],    # To Change direction of x axis from low to high 
                                 mode='lines', name='Diff',
                                 line=dict(color=self.colors[1])
                                 ))
        # Configrue the Layout
        fig.update_layout(
            title='Noise',
            xaxis_title='Chemical shift [ppm]',
            yaxis_title='Intensity',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(
                range=[self.differences.iloc[:,frame].min()*20, self.max_y],
                titlefont=dict(size=18),          # Font size for y-axis title
                tickfont=dict(size=18)
                       ),
            xaxis=dict(
                range=[self.max_x, self.min_x], 
                dtick=0.5,
                titlefont=dict(size=18),          # Font size for y-axis title
                tickfont=dict(size=18),
                ),                                                                                                                          # To Change direction of x axis from low to high 
            template=self.template
            )

        return fig

    def plot_sum_fit(self, frame):
        """
        Generates a plot for the sum fit and individual Lorenzian fits for a given frame.

        This method creates a plot that displays the sum of fitted Lorentzian curves as well as individual Lorentzian fits for the specified frame. The x-axis represents the chemical shift (in ppm), and the y-axis represents the intensity. Each individual Lorentzian fit is plotted in a different color, and the sum of all Lorentzian curves is shown as a separate trace.

        Args:
            frame (int): The frame index for which the sum fit and individual Lorentzian fits will be generated. It corresponds to the column in both the sum data and individual fit data.

        Returns:
            plotly.graph_objects.Figure: The figure containing the sum fit and individual Lorentzian fits for the specified frame.

        Attributes Set:
            min_y (float): The minimum y-value across all data for consistent y-axis scaling.
            max_y (float): The maximum y-value across all data for consistent y-axis scaling.
            min_x (float): The minimum x-value for the x-axis based on the data.
            max_x (float): The maximum x-value for the x-axis based on the data.
            colors (list): A list of colors used to differentiate the individual fits and the sum fit.
        """

        frame_data = self.individual_fits[frame -1 ]

        # Add sum fit data
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=self.sum_data.iloc[:,0][::-1],           # To Change direction of x axis from low to high 
                                 y=self.sum_data.iloc[:,frame][::-1],       # To Change direction of x axis from low to high 
                                 mode='lines', 
                                 name='Sum Fit',
                                 line=dict(color=self.colors[2])
                                 ))
        
        # add sum fit for each frame
        for i in range(1, len(frame_data.columns)):
            color = self.colors[(i + 2) % len(self.colors)]
            fig.add_trace(go.Scatter(x=self.sum_data.iloc[:,0], 
                                     y=frame_data.iloc[:,i], 
                                     mode='lines', 
                                     name=frame_data.columns[i],
                                     line=dict(color=color)
                                     ))

        # configure the Layout
        fig.update_layout(
            title='Fitted Lorenzian Curves',
            xaxis_title='Chemical shift [ppm]',
            yaxis_title='Intensity',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(
                range=[self.min_y, self.max_y],
                titlefont=dict(size=18),          # Font size for y-axis title
                tickfont=dict(size=18),
                ),
            xaxis=dict(
                range=[self.max_x, self.min_x], 
                dtick=0.5,
                titlefont=dict(size=18),          # Font size for y-axis title
                tickfont=dict(size=18)),                      # To Change direction of x axis from low to high 
            template=self.template
            )

        return fig
    
    def save_fig(self, fig, name):
        """
        Saves a plotly figure to both PDF and PNG formats.

        This method saves the given figure to disk in both PDF and PNG formats using the Kaleido engine. The file names are specified by the provided `name` argument, with appropriate extensions (.pdf and .png) added.

        Args:
            fig (plotly.graph_objects.Figure): The plotly figure to be saved.
            name (str): The base name to be used for the saved files (without extensions). The files will be saved as <name>.pdf and <name>.png.

        Returns:
            None

        Note:
            The saved images will have a width of 1200 pixels and a height of 800 pixels.
        """

        # Save to PDF
        pio.write_image(fig, f'{name}.pdf', format='pdf', engine='kaleido', width=1200, height=800)

        # Save to PNG
        pio.write_image(fig, f'{name}.png', format='png', engine='kaleido', width=1200, height=800) 
        


   