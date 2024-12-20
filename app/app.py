import Layout
import panel3_contour_plot as p3cp
from pathlib import Path
import pandas as pd
from LoadData import *
import streamlit as st




def main():
    """
    Main function to initialize and run the Streamlit app.

    This function creates an instance of the `StreamlitApp` class from the `Layout` module, which is responsible for setting up the layout and user interface of the application. After the instance is created, the `run` method is called to start the application.

    Args:
        None

    Returns:
        None
    """

    app = Layout.StreamlitApp()
    app.run()


if __name__ == '__main__':
    main()

