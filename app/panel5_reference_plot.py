import pandas as pd
import matplotlib.pyplot as plt
import re
import peak_fitting_v6
import os
from pathlib import Path
import numpy as np


class Reference():
    def __init__(self, fp_ref, fp_meta, fp_data):
        self.data = pd.read_csv(fp_ref)
        self.chem_shifts = self.data.iloc[:,0]
        self.LorentzianFit = peak_fitting_v6.PeakFitting(fp_file = fp_ref , fp_meta = fp_meta)
        self.fitting_params = self.LorentzianFit.fit(save_csv= False)
        self.reference_value = self.ReferenceValue()
        self.file_name_ref = os.path.basename(fp_ref)
        self.plot_dir = Path('output', self.file_name_ref + '_output', 'plots')
        self.reference_pdf = Path(self.plot_dir, f'Reference_{self.file_name_ref}')
        #kinetics
        self.file_name = os.path.basename(fp_data)
        self.output_dir = Path('output', self.file_name + '_output')

        self.kin_fp = Path('output', os.path.basename(fp_data) + '_output', 'kinetics.csv')
        self.kin_df = pd.read_csv(self.kin_fp)

        # Ensure the plot directory exists 
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def ReferenceValue(self):
        """ne

        Returns:
            reference_value: factor to calculate concentration in mmol from integral value
        """
        #get reference concentration from meta data
        mmol = re.findall(r'[0-9\.]+', self.LorentzianFit.meta_df.iloc[0]['Substrate_mM_added'])
        mmol = float(mmol[0])

        if mmol:
            print()  
        else:
            print("mMol value couldn't be extracted from Substrate_mM_added ")

        # calculate ref_factor
        reference_value = mmol / self.fitting_params['Water_amp_4.7'].mean()

        return reference_value
 
    
    def plot(self, i):
        """_summary_

        Args:
            i (_type_): _description_

        Returns:
            _type_: _description_
        """
        spectra_data = self.data.iloc[:,i+1]


        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # amplitude
        ax[0].plot(self.fitting_params['Water_amp_4.7']*np.pi)
        ax[0].axhline(y=(self.fitting_params['Water_amp_4.7']*np.pi).mean(), color='grey', linestyle='--')
        ax[0].set_title('Integral of water over time')
        ax[0].set_xlabel('Time step') 
        ax[0].set_ylabel('Integral value water peak')
        # annotation
        ax[0].annotate(f'Calculated Convergence Factor = {self.reference_value/np.pi:.4f}', # match with integral
                       xy=(1.05, 0.85), xycoords='axes fraction', 
                       xytext=(-20, 20), 
                       textcoords='offset points',
                       fontsize = 8, 
                       ha='right', 
                       va='top')

        #axs[0].legend()

        # Second plot
        #actual curve
        ax[1].plot(self.chem_shifts, spectra_data, c='blue', label='Reference spectrum')
        
        # Lorentzian
        y_lorentzian = self.LorentzianFit.lorentzian(x=self.data.iloc[:,0], 
                                                 shift= self.fitting_params.iloc[i]['Water_pos_4.7'],
                                                  gamma= self.fitting_params.iloc[i]['Water_width_4.7'], 
                                                 A= self.fitting_params.iloc[i]['Water_amp_4.7'])
        ax[1].plot(self.chem_shifts, y_lorentzian + self.fitting_params.iloc[i]['y_shift'] , c='red', label='Lorentzian fit')
        
        
        ax[1].set_xlabel('Chemical shift [ppm]')
        ax[1].set_ylabel('Intensity')
        ax[1].set_title(f'Lorentzian fit for time step: {i}')
        ax[1].set_xlim(max(self.chem_shifts),min(self.chem_shifts))
        ax[1].legend()

        # global title
        fig.suptitle('Reference spectrum and Lorentzian fit of File: ' + self.file_name_ref)
        plt.tight_layout()
    
        return fig  
    
    def save_fig(self, fig, name, width=1200, height=800):
        """_summary_

        Args:
            fig (_type_): _description_
            name (_type_): _description_
            width (int, optional): _description_. Defaults to 1200.
            height (int, optional): _description_. Defaults to 800.
        """
        
        #Konvertieren der Breite und Höhe von Pixel in Zoll (dpi = 300)
        fig.set_size_inches(width / 100, height / 100)
        fig.savefig(f'{name}.pdf', format='pdf')
        fig.savefig(f'{name}.png', format='png')

    def save_kinetics_mmol(self):
        """_summary_
        """
        kin_mmol = self.kin_df.copy().set_index('Time_Step') 
        value_col = ['ReacSubs','Metab1','Water'] 

        for col in value_col:
            kin_mmol[col] *=self.reference_value

        kin_mmol.to_csv(Path(self.output_dir, 'kinetics_mmol.csv'))

    
