import streamlit as st
import pandas as pd
import os
from pathlib import Path
from LoadData import *
from panel3_contour_plot import *
from Process4Panels import Process4Panels
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from panel1_spectrum_plot import Panel1SpectrumPlot
from panel2_kinetic_plot import KineticPlot
from panel3_contour_plot import ContourPlot
from panel5_reference_plot import Reference


# set page config
st.set_page_config(layout="wide", page_title="SBMI - Application", page_icon=":shark:")

def select_file(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]):
    """Load data path from file dialog.

    Returns:
        dir_path (str): directory path
    """
    root = Tk()
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    #dir_path = r'/Users/marco/Documents/UKSH/Data/Banzhaf_Marco_1996-09-03_B.MA._2/2024-04-11'
    root.destroy()
    if file_path == '':
        return None
    return file_path

class StreamlitApp():
    """
    A Streamlit-based application for visualizing and analyzing metabolite spectra and kinetics.

    This application provides a user-friendly interface for uploading CSV files, visualizing
    plots such as substrate plots, kinetic plots, contour plots, and reference plots, and 
    navigating between panels for different types of analyses.

    Attributes:
        fig1 (str or plotly.graph_objects.Figure): Spectrum plot or its file path.
        fig2 (str or plotly.graph_objects.Figure): Kinetic plot or its file path.
        fig3 (str or plotly.graph_objects.Figure): Contour plot or its file path.
        fig4 (str or plotly.graph_objects.Figure): Reference plot or its file path.

    
    Methods:
        side_bar():
            Creates a sidebar in the Streamlit app with instructions for usage.
        
        header():
            Displays the application header and file uploader, and initializes navigation tabs.

        main_page(main):
            Manages the main content of the application, displaying different panels for analysis.

        about_page(about):
            Displays the "About" section of the application with descriptive text.

        panel1():
            Displays the content for Panel 1, focusing on the substrate plot.

        panel2():
            Displays the content for Panel 2, focusing on the kinetic plot.

        panel3():
            Displays the content for Panel 3, allowing for further exploration.

        panel4():
            Displays the content for Panel 4, offering additional visualizations or data.

        run():
            Executes the Streamlit application, starting from the header and displaying all panels.
    """

    def __init__(self, 
                 fig1=None, 
                 fig2=None,
                 fig3=None,
                 fig4=None
                 ):
        
        self.fig1 = fig1
        self.fig2 = fig2
        self.fig3 = fig3
        self.fig4 = fig4

# /home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx
# /home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/FA_20240731_2H_yeast_Fumarate-d2_15_200.ser.csv

# '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx'
# '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/FA_20240207_2H_yeast_Fumarate-d2_5.csv'



    def header(self):
        st.markdown("""<h1 style="text-align: center;">SBMI - Application</h1>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.2, 0.8, 0.2])
        with col1:
            st.divider()

        with col2:
            
            # Selector for choosing the model
            options = ["Model 1: Lorentzian Fit", "Model 2: Lorentzian fit with prefitting"]
            selected_option = st.selectbox("Choose the Model:", options)

            # Dynamically load and store the class in session_state
            if selected_option == "Model 1: Lorentzian Fit":
                
                #if 'Model 1' in st.session_state:
                st.session_state['Model 1'] = True  # Store the class in session state
                st.session_state['Model 2'] = False 
            
            elif selected_option == "Model 2: Lorentzian fit with prefitting":
                
                #if 'Model 2' in st.session_state:
                st.session_state['Model 1'] = False  # Store the class in session state
                st.session_state['Model 2'] = True


            sub_col1, sub_col2 = st.columns([0.30, 0.70])

            # Select the Metadata as xml
            with sub_col1:
                st.markdown('**Step 1: Select the Metadata as .xlsx**')
                select_meta_button = st.button(label='Select Metafile')
                if select_meta_button:
                    self.meta_fp = select_file(filetypes=[("XLSX files", "*.xlsx")])
                    st.session_state["meta_file"] = self.meta_fp
                
            # Select the Spectrum as csv 
            with sub_col1:
                st.markdown('**Step2: Select Spectrum as .csv**')
                select_file_button = st.button(label='Select Spectrum', )
                if select_file_button:
                    self.data_fp = select_file(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                    st.session_state["file_name"] = self.data_fp

            # Select the Reference file as 
            with sub_col1:
                st.markdown('**Step3: Select the Reference File as .csv**')
                select_reference_button = st.button(label='Select Reference')
                if select_reference_button:
                    self.reference_fp = select_file(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                    st.session_state['reference_file'] = self.reference_fp
                    

        # Session State for Model import
        if st.session_state['Model 1'] == True:
            from peak_fitting_v6 import PeakFitting
        else:
            from peak_fitting_v7 import PeakFitting

        with sub_col2:
            # Session State for files
            if 'meta_file' in st.session_state:
                self.meta_fp = st.session_state['meta_file']
                st.write('')
                st.info(self.meta_fp)
            else:
                st.warning("No meta file selected or key does not exist.")
            if "file_name" in st.session_state:
                self.data_fp = st.session_state["file_name"]
                with sub_col2:
                        st.write('')
                        st.info(self.data_fp)
            else:
                st.warning("No substrate file selected or file does not exist.")
                
            if 'reference_file' in st.session_state:
                self.reference_fp = st.session_state['reference_file']
                with sub_col2:
                    st.write('')
                    st.info(self.reference_fp)
                
            else:
                st.warning("No reference file selected or key does not exist.")

        with col2:
            process_col1, process_col2, process_col3 = st.columns([2, 1, 1])  # 1:2:1 ratio
        with process_col1:
            st.markdown('**Step 4: Press Start Processing**')
        with process_col2:
            if st.button("Start Processing"):
                    st.session_state['button_pressed'] = True
                    st.session_state["processing_started"] = True

        if 'processing_started' in st.session_state :
            if 'button_pressed' in st.session_state:
                if st.session_state['button_pressed']:
                    information = st.empty()
                    information.info("Processing the data. Please wait...")
                    self.process_data(PeakFitting)
                    self.process_plots()
                    information.empty()
                    st.session_state['button_pressed'] = None

                    
        with col3:
            st.divider()
        # Slit into Main and Instruction Tabs
        main, about = st.tabs(['Main Page', 'Instructions'])


        # Dynamically run main_page only if the button is clicked
        if 'processing_started' in st.session_state:
            if st.session_state["processing_started"]: # Set to false if it should open after pressing the button
                if st.session_state["file_name"] is not None:
                    self.main_page(main)
        self.about_page(about)
        
    def main_page(self, main):
        with main:
            st.markdown("#### Main Page Content")
            if st.session_state.get("processing_started", True): # Set to false if it should open after pressing the button
                self.panel1()
                self.panel2()
                self.panel3()  
                self.panel4()

                # Shutdown button
                # exit_app = st.button("Shut Down")
                # if exit_app:
                #     Give a bit of delay for user experience
                #     time.sleep(0.1)
                #     Close streamlit browser tab
                    
                #     keyboard = Controller()
                #     Simulate pressing and releasing "ctrl+w"
                #     keyboard.press(Key.ctrl)
                #     keyboard.press('w')
                #     keyboard.release('w')
                #     keyboard.release(Key.ctrl)

                #     Terminate streamlit python process
                #     pid = os.getpid()
                #     p = psutil.Process(pid)
                #     p.terminate()
            else:
                st.info("Click 'Start Processing' to see the analysis panels.")
    
    def process_data(self, PeakFitting):
        #perform peak fitting
        fitter = PeakFitting(self.data_fp, self.meta_fp)
        fitter.fit()
        #Create data for the panels
        processor = Process4Panels(self.data_fp)
        processor.save_sum_spectra()
        processor.save_substrate_individual()
        processor.save_difference()
        processor.save_kinetics()
    
    def process_plots(self):
        st.session_state['panel_1_obj'] = Panel1SpectrumPlot(file_path = self.data_fp)
        st.session_state['panel_2_obj'] = KineticPlot(self.data_fp)
        st.session_state['panel_3_obj'] = ContourPlot(self.data_fp)
        st.session_state['panel_obj_4'] = Reference(fp_ref = self.reference_fp, fp_meta = self.meta_fp, fp_data = self.data_fp)

        st.session_state['panel_obj_4'].save_kinetics_mmol()

    def about_page(self, about):
        with about:
            # Text which will be shown in the Instruction 
            st.markdown(f"""
                        ### Instructions:

                        #### Step 0: 
                        **Choose the Model:**

                        **Model 1:**

                        Lorenzian curve fitting with parameters of the Meta Description

                        **Model 2:**

                        Lorenzian curve fitting + initial parameters derived from actual spectrum

                        #### Step 1:
                        - **Select the Metadata:** 
                            
                        The Metadata should be an xlsx file that has the following structure:
                        | ID | File                                     | Expt_name                   | TR[s] | NS | TR total [s] | Substrate_name       | Substrate_N_D | Substrate_mM | Substrate_ppm | pH_before | pH_after | Reaction_temperature (Kelvin) | Yeast_suspension      | Substrate_solvent | Substrate_mM_added | Water_ppm | Metabolite_1      | Metabolite_2 | Metabolite_3 | Metabolite_4 | Metabolite_5 | Metabolite_1_ppm | Metabolite_2_ppm | Metabolite_3_ppm | Metabolite_4_ppm | Metabolite_5_ppm |
                        |----|-----------------------------------------|-----------------------------|-------|----|--------------|----------------------|---------------|--------------|---------------|-----------|----------|-------------------------------|-----------------------|-------------------|--------------------|-----------|-------------------|--------------|--------------|--------------|--------------|------------------|------------------|------------------|------------------|------------------|
                        | 1  | FA_20240206_2H_yeast_Acetone-d6_3.csv   | FA_20240206_2H_yeast_1_3    | 17.50 | 8  | 140          | Acetone-d6           | 6             | 15           | 2.32          | 5.06      | 310      | 1g yeast in 7mL water        | PBS (50mM)           | 30mM              | 4.70     | Propan-2-ol-d6 |              |              |              |              | 1.20             |                  |                  |                  |                  |
                        | 2  | FA_20231123_2H_Yeast_Fumarate-d2_12.csv | FA_20231123_2H_Yeast_1_12   | 11.50 | 8  | 92           | Fumarate-d2          | 1,1           | 15           | 6.65          | 5.62      | 310      | 1g yeast in 7mL water        | PBS (50mM)           | 30mM              | 4.70     | Malate-d2(sum) |              |              |              |              | 4.368, 2.474     |                  |                  |                  |                  |
                        | 3  | FA_20240207_2H_yeast_Fumarate-d2_5.csv  | FA_20240207_2H_yeast_1_5    | 11.50 | 8  | 92           | Fumarate-d2          | 1,1           | 15           | 6.65          | 5.60      | 310      | 1g yeast in 7mL water        | PBS (50mM)           | 30mM              | 4.70     | Malate-d2(sum) |              |              |              |              | 4.368, 2.474     |                  |                  |                  |                  |


                        - **Select the Substrate**

                        The subtrate file should be a .csv file which have the folloing structure:
                        | 2H Chemical Shift (ppm) | FA_20240105_2H_yeast_1.5.ser#1 | FA_20240105_2H_yeast_1.5.ser#2 | FA_20240105_2H_yeast_1.5.ser#3 | FA_20240105_2H_yeast_1.5.ser#4 |
                        |--------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
                        | Value 1                 | X1                             | X2                             | X3                             | X4                             |
                        | Value 2                 | X5                             | X6                             | X7                             | X8                             |
                        | Value 3                 | X9                             | X10                            | X11                            | X12                            |
                        | ...                     | ...                            | ...                            | ...                            | ...                            |
                        
                        where X is the measured value


                        - **Select the Reference File:**
                        The reference file should be a .csv file which has the following structre:

                        | 2H Chemical Shift (ppm) | FA_20240105_2H_yeast_1.5.ser#1 | FA_20240105_2H_yeast_1.5.ser#2 | FA_20240105_2H_yeast_1.5.ser#3 | FA_20240105_2H_yeast_1.5.ser#4 |
                        |--------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
                        | Value 1                 | X1                             | X2                             | X3                             | X4                             |
                        | Value 2                 | X5                             | X6                             | X7                             | X8                             |
                        | Value 3                 | X9                             | X10                            | X11                            | X12                            |
                        | ...                     | ...                            | ...                            | ...                            | ...                            |
                        
                        where X is the measured value


                        #### Step 2:
                        - Click Start Processing

                        #### Step 3:
                        ##### Substrate Plot
                        - Use Sliders to slect the Frame, to investigate the Spectrum which its coresponding fitted metabolite and substrates
                        - In the legend, select the line to be or not to be visualized

                        #### Kinetic Plot:
                        - Kinetcs of the metabolites and substrate peaks
                        
                        ##### Contour Plot
                        - Visualize the full measured spectrum and select the depth [%] to visualize peaks


                        #### Reference Plot
                        - Get the reference value on water.


                        """)
        return None

    # -------- Panels with Expanders --------------------
    def panel1(self):
        # read in results from fitting
        try:
            sum_fit_fp = Path('output', f'{os.path.basename(self.data_fp)}_output', 'sum_fit.csv')
        
            sum_fit = pd.read_csv(sum_fit_fp)
        except:
            # Show this text message if there the Output Data is no available.
            st.markdown("""
                        <span style="color:red; font-size:72px;">Please Press 'Start Processing !'</span>
                        """, unsafe_allow_html=True)

        with st.expander("Panel 1 - Substrate Plot", expanded=True):
            # add a slider to select the frame
            st.session_state['time_frame'] = st.slider('Select the frame', min_value=1, max_value=sum_fit.shape[1], value=1)
            st.markdown('# Substrate Plot')
            st.write(f"Standard deviation of noise: {st.session_state['panel_1_obj'].differences.iloc[:,st.session_state['time_frame']].std():.3f}")
            #panel_1_obj = Panel1SpectrumPlot(file_path = self.data_fp)
            one_plot = st.session_state['panel_1_obj'].plot(st.session_state['time_frame'])
            st.plotly_chart(one_plot, use_container_width=True)

            # Save the Plot as PDF
            save_contour_button = st.button(label='Save Substrate as PDF')
            if save_contour_button:
                self.save_to_pdf(session_state=st.session_state['panel_1_obj'],
                                fig=one_plot,
                                file_basename=os.path.basename(self.data_fp),
                                file_name=f"Substrate_{os.path.basename(self.data_fp)}_{st.session_state['time_frame']}"                             )

            
    def panel2(self):
        """ Kinetic Plot"""
        with st.expander("Panel 2 - Kinetic Plot", expanded=True):
            st.markdown('# Kinetic Plot')
            fig = st.session_state['panel_2_obj'].plot() 
            st.plotly_chart(fig, use_container_width=True)

            # Save the Plot as PDF
            save_contour_button = st.button(label='Save Kinectic as PDF')
            if save_contour_button:
                self.save_to_pdf(session_state=st.session_state['panel_2_obj'],
                                fig=fig,
                                file_basename=os.path.basename(self.data_fp),
                                file_name=f'Kinetic_{os.path.basename(self.data_fp)}'
                                )

    def panel3(self):
        """Contour Plot"""
        with st.expander("Panel 3 - Contour Plot", expanded=True):
            st.markdown('# Contour Plot')
            # one range slider for both max and min
            zmin_zmax = st.slider('Select Zmin and Zmax', min_value=0.0, max_value=1.0, value=(0.0, 1.0))
            contourplot = st.session_state['panel_3_obj'].plot(zmin=zmin_zmax[0], zmax=zmin_zmax[1])
            st.pyplot(contourplot, clear_figure=False)

            # Save the Plot as PDF
            save_contour_button = st.button(label='Save Contour as PDF')
            if save_contour_button:
                self.save_to_pdf(session_state=st.session_state['panel_3_obj'],
                                fig=contourplot,
                                file_basename=os.path.basename(self.data_fp),
                                file_name=f'Contour_{os.path.basename(self.data_fp)}_{zmin_zmax}'
                                )

    def panel4(self):
        """Reference Plot"""
        with st.expander("Panel 4 - Reference", expanded=True):
            st.markdown('# Reference')
            i = st.slider('Select the frame for water reference', min_value=1, max_value= st.session_state['panel_obj_4'].data.shape[1], value=1) #max_value ist falsch, sessionstate?
            reference_plot = st.session_state['panel_obj_4'].plot(i = i)
            st.pyplot(reference_plot)

            # Save the Plot as PDF
            save_reference_button = st.button(label='Save Reference as PDF')
            if save_reference_button:
                self.save_to_pdf(session_state=st.session_state['panel_obj_4'],  # Stellen Sie sicher, dass panel_3_obj das korrekte Objekt für diesen Aufruf ist
                        fig=reference_plot,
                        file_basename=os.path.basename(self.reference_fp),  # Verwendung des Basisnamens von data_fp, wie im funktionierenden Beispiel
                        file_name=f'Reference_{os.path.basename(self.reference_fp)}_{i}')
                                    
                
    # os.path.basename(fp_ref)
    # self.plot_dir = Path('output', self.file_name + '_output', 'plots')
    # self.reference_pdf = Path(self.plot_dir, f'Reference_{self.file_name}'
    def save_to_pdf(self, session_state, fig, file_basename, file_name):
        
        plot_dir = Path('output', file_basename + '_output', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        name = Path(plot_dir, f'{file_name}_{file_basename}')

        session_state.save_fig(fig=fig, name=name)

        

    def run(self):
        self.header()
