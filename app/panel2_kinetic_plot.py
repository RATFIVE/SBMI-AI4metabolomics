import pandas as pd
import plotly.express as px
from pathlib import Path
import os
import plotly.io as pio
import numpy as np

class KineticPlot:
    """
    KineticPlot Class

    The KineticPlot class is designed to handle the visualization and saving of kinetic data from a CSV file. It takes in a file path, reads the corresponding kinetic data, and creates a plot of substance kinetics over time, with an option to save the generated figure to disk.

    Attributes:
        path (str): The file path to the data source.
        basename (str): The base name of the file, extracted from the given path.
        plot_dir (Path): The directory where plot images will be saved.
        kin_fp (Path): The path to the kinetic data CSV file.
        kin_df (pandas.DataFrame): The dataframe containing the kinetic data.
        kinetic_pdf (Path): The path where the plot will be saved as a PDF.
        colors (list): A list of color options for the plot, using Plotly's Dark24 color palette.
        template (str): The template to use for the plot's layout (set to 'plotly_white').

    Methods:
        __init__(self, path):
            Initializes the KineticPlot object with the given file path, processes the CSV data, and sets up directories and paths for saving plots.

        plot(self):
            Creates a scatter plot using the kinetic data, where each series (column) is plotted over time steps. The data is multiplied by pi to represent the integral of the kinetic values. It customizes the layout and appearance of the plot, including title, axis labels, and legend.

            Returns:
                plotly.graph_objects.Figure: The generated scatter plot figure.

        save_fig(self, fig, name):
            Saves the given figure as both a PDF and PNG file using the Kaleido engine. The files are saved with the provided base name and appropriate extensions (.pdf and .png).

            Args:
                fig (plotly.graph_objects.Figure): The plotly figure to be saved.
                name (str): The base name to be used for the saved files (without extensions).
            
            Returns:
                None
    """

    def __init__(self, path):   
        self.path = path
        # init path
        self.basename = os.path.basename(self.path)
        self.plot_dir = Path('output', self.basename + '_output', 'plots')
        self.kin_fp = Path('output', os.path.basename(self.path) + '_output', 'kinetics.csv')
        self.kin_df = pd.read_csv(self.kin_fp)
        self.kinetic_pdf = Path(self.plot_dir, f'Kinetic_{self.basename}')
        self.colors = px.colors.qualitative.Dark24
        self.template = 'plotly_white'
 
    def plot(self):
        """
        plot(self)

        Generates a scatter plot for the substance kinetics data stored in the `kin_df` dataframe. The plot displays the integral of the kinetic values, calculated by multiplying the data by pi. Each column (except for the time step) is plotted as a separate series over time, with the x-axis representing the time steps and the y-axis representing the integral of the values. The plot is customized with labels, colors, and layout settings.

        Returns:
            plotly.graph_objects.Figure: The generated scatter plot figure.

        Attributes:
            kin_df (pandas.DataFrame): The dataframe containing the kinetic data, with the first column being the time steps and the subsequent columns representing the kinetic values.
            colors (list): A list of colors used for the plot, starting from the fourth color in the `Dark24` color palette.
            basename (str): The base name of the data file, used to create the plot's title.
            template (str): The plot template to apply to the layout (set to 'plotly_white').

        Example:
            kin_plot = KineticPlot(path="path/to/data.csv")
            fig = kin_plot.plot()
        """

        # use the colors from the 3 position
        colors = self.colors[3:]
        # add kinetic data
        fig = px.scatter(
            self.kin_df * np.pi, # Multiply by pi to get the integral
            x='Time_Step',
            y=self.kin_df.columns[1:],  # Select all columns except 'time' for y
            labels={'value': 'Value', 'variable': 'Series'},
            color_discrete_sequence=colors
        )

        # configure the layout
        fig.update_layout(
            title=dict(
                text=f'Substance Kinetics for File {self.basename}',
                font=dict(size=24)  # Font size for the title
            ),
            xaxis_title='Time step',
            yaxis_title='Integral',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(
                titlefont=dict(size=18),          # Font size for y-axis title
                tickfont=dict(size=18)            # Font size for y-axis ticks
            ),
            xaxis=dict(
                titlefont=dict(size=18),          # Font size for x-axis title
                tickfont=dict(size=18)            # Font size for x-axis ticks
            ),
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
        