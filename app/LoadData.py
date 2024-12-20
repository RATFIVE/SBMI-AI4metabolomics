import pandas as pd
import os
from pathlib import Path
import math
import numpy as np


class LoadData:
    def __init__(self):
        """
        Initializes the LoadData class.
        
        This class provides methods for loading data files and retrieving
        information related to specific files, including data descriptions
        and substrate lists.
        """
        pass

    def load_data_list(self, endswith:str):
        """
        Loads all files from the 'Data' directory that have the specified file extension.

        This function searches through the 'Data' directory (one level up from the
        current working directory) for files that match the specified file extension.
        It collects the paths of these files and returns them as a list.

        Example files:
            - 'FA_20231113_2H_yeast_Pyruvate-d3_1.csv'

        Args:
            endswith (str): The file extension to filter by (e.g., '.csv').

        Returns:
            list: A list of file paths in the 'Data' directory that end with the specified extension.
        """
        path_list = []
        cwd = Path(os.getcwd())

        data_path = os.path.join(cwd.parent, 'Data')

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(endswith):
                    file_path = os.path.join(root, file)
                    path_list.append(file_path)
        return path_list

    def load_DataDescription(self):
        """
        Loads the 'DataDescription.csv' file as a pandas DataFrame.

        This function reads 'DataDescription.csv' from the 'Data' directory located
        one level up from the current working directory. It returns the file's content
        as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the data from 'DataDescription.csv'.
        """
        data_description_path = os.path.join(os.getcwd(), '..', 'Data', 'Data_description_main.xlsx')
        data_description = pd.read_excel(data_description_path, engine='openpyxl')
        #display(data_description)
        return data_description


    def get_substrate_list(self, file: str):
        """
        Retrieves substrate information related to the specified file.

        This function loads data from the 'DataDescription.csv' file, filtering
        rows based on the specified file name. It then extracts the substrate's
        chemical shift (ppm) and water chemical shift (ppm) values, returning them
        as a list.

        Args:
            file (str): The name of the file for which to retrieve substrate information.

        Returns:
            list: A list containing the substrate chemical shift (ppm) and water chemical shift (ppm) as floats.
        """
        data_desc = self.load_DataDescription()

        # Filter by file name
        data_desc = data_desc.loc[data_desc['File'] == file].reset_index(drop=True)

        substrat_shift = data_desc.at[0, 'Substrate_ppm']
        substrat_water = data_desc.at[0, 'Water_ppm']
        
        def clean_list(value):
            # Convert value to float
            return float(value)
        

        # Ensure substrat_shift is treated as a list
        if isinstance(substrat_shift, (float)):
            substrat_shift = [substrat_shift]
        else:
            substrat_shift = [clean_list(x) for x in substrat_shift.strip().split(',')]
        
        # Ensure substrat_water is treated as a list
        if isinstance(substrat_water, (float)):
            substrat_water = [substrat_water]
        else:
            substrat_water = [clean_list(x) for x in list(substrat_water)]

        # Return as list
        substrates = substrat_shift + substrat_water 
   
        return substrates

    def get_metabolite_list(self, file):
        """
        Retrieves a list of metabolite chemical shifts (ppm) associated with the specified file.

        This method loads data from the 'DataDescription.csv' file and filters it by the specified file name.
        It then extracts all columns containing metabolite information (columns with 'Metabolite' and 'ppm' in
        the name) and returns their values as a list of floats, excluding any NaN values.

        Args:
            file (str): The name of the file for which to retrieve metabolite chemical shift information.

        Returns:
            list: A list of metabolite chemical shift values (in ppm) as floats, excluding NaN values.
    """
        
        data_desc = self.load_DataDescription()
        
        # Filter by file name
        data_desc = data_desc.loc[data_desc['File'] == file].reset_index(drop=True)

        # get all cols which contains Metabolite and ppm
        cols = [col for col in data_desc.columns if 'Metabolite' in col and 'ppm' in col]
        metabolites = []
        for col in cols:
            metabolites.append(data_desc.at[0, col])

        # to float
        if isinstance(metabolites[0], str):
            metabolites = [float(metabolite) for metabolite in metabolites[0].strip().split(',')]
        else:
            metabolites = [float(metabolite) for metabolite in metabolites if not math.isnan(metabolite)]
        
        # remove nan values
        metabolites = [metabolite for metabolite in metabolites if not math.isnan(metabolite)]
        
        return metabolites
    

if __name__ == '__main__':

    loaddata = LoadData()

    df_list = loaddata.load_data_list('FA_20231123_2H Yeast_Fumarate-d2_12 .csv')
    df = pd.read_csv(df_list[0])