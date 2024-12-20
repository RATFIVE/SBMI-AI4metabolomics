import pandas as pd
import numpy as np


class SpectraAnalysis:
    """
    A class to locate Peaks based on the second derivative previously to running the Lorentzian fit in Model 2
    """
    def __init__(self, data, expected_peaks):
        self.data = data
        self.expected_peaks = expected_peaks
        self.spectra_data = data.iloc[:, 1:]
        self.chem_shifts = data.iloc[:, 0]

    def peakfit_sum(self, threshold):
        """sums up all spectra and returns list of found peaks above threshold percentile

        Args:
            spectra_data(DataFrame): extracted intensitys from DataFrame data
            chem_shifts(list): extracted from DataFrame data
        
        Returns:
            peak_pos(list): list with found peak positions
        """
        sum_of_spectra = np.sum(self.spectra_data, axis=1)
        threshold = np.percentile(sum_of_spectra, threshold)

        first_derivative = np.gradient(sum_of_spectra, self.chem_shifts)
        second_derivative = np.gradient(first_derivative, self.chem_shifts)
        third_derivative = np.gradient(second_derivative, self.chem_shifts)
        fourth_derivative = np.gradient(third_derivative, self.chem_shifts)
        
        #calculate minima below zero in 2nd derivative
        sign_change = np.diff(np.sign(third_derivative)) != 0
        peak_mask = (sum_of_spectra > threshold) & (second_derivative < 0) & (fourth_derivative > 0)
        peak_mask[1:] &= sign_change

        #return list
        peak_pos = self.chem_shifts[peak_mask].tolist()
        return peak_pos

    def normalize_water(self):
        """shifts chem_shift values based on summed up spectra, so water peak is normalized to 4.7

        Args:
            spectra_data(DataFrame): extracted intensitys from DataFrame data
            chem_shifts(list): extracted from DataFrame data
        
        Returns:
            data_normalized(DataFrame): DataFrame with normalized Data   
            norm_shift(Float): value by which chem_shift was shifted

        """    
        #85th percentile as threshold
        peak_pos = self.peakfit_sum(threshold = 85)
        water = 4.7

        #select peal closest to 4.7 and calculate shift
        closest_peak = min(peak_pos, key=lambda x: abs(x - water))
        norm_shift = (4.7 - closest_peak)

        #shift spectra
        data_normalized = pd.DataFrame(self.chem_shifts.copy() + norm_shift)
        data_normalized = pd.concat([data_normalized, self.data.iloc[:, 1:]], axis=1)
        data_normalized.columns = self.data.columns
        return data_normalized, norm_shift

    def peak_identify(self, data_normalized, reference_peaks, initial_threshold=85, max_shift=0.1): #max_shift maybe
        """Searches for peaks and adds them to list based on expected values. 
        Threshold is continually lowered until all expected peaks are found or more unknown peaks are found than expected
        
        Args:

            self.expected_peaks(list): chem shift values of expected peaks
            data_normalized(DataFrame): data normalized, result of function normalize_water
            reference_peaks(list) = usually self.expected_peaks
            initial_threshold(int, optional): starting percentile above which peaks are recognizedDefaults to 85
            max_shift(Float): maximum value that chem_shift can be shifted from expected peakposition for identification

        Returns:

            found(list): list of actual peak positions in order of expected peaks
            other(list): list of unidentified found peaks
        """
        spectra_data = data_normalized.iloc[:, 1:]
        chem_shifts = data_normalized.iloc[:, 0]

        found = [None] * len(reference_peaks)
        other = []
        threshold = initial_threshold

        while None in found or len(other) <= len(found):
            #search for peaks
            detected_peaks = self.peakfit_sum(threshold)

            #match found peaks to expected peaks
            for peak in detected_peaks:
                distances = [abs(peak - expected_peak) for expected_peak in reference_peaks]
                min_distance = min(distances)
                index = distances.index(min_distance)

                if min_distance <= max_shift:
                    if found[index] is None:
                        found[index] = peak
                    elif found[index] != peak:
                        other.append(peak)
                else:
                    other.append(peak)

            #lower threshold 
            threshold -= 2
            if threshold < 0 or None not in found or len(other) > len(found):
                break
        
        # replace remaining unfound peak positions with expected chem_shift value 
        for i, peak in enumerate(found):
            if peak is None:
                found[i] = reference_peaks[i]  


        return found, other

    def peak_df(self, min_cols_per_section=5, max_shift = None):
        """Normalizes the given data to chem_shift(water) = 4.7 
        splits spectra into sections over time
        identifies peaks in summed up sections

        Args:
            data(dataFrame): spectrum data; 1.column with chemical shift, athers with intensity
            expected_peaks (list): list of chem_shifts of expected peaks
            min_cols_per_section (int, optional): minimum number if spectra that are summed up to find peaks. Defaults to 20.
            max_shift(Float, optional) = max shift in chem_shift of peaks between section. If not given, calculated based on assumption of max total shift during experiment =1. 
                Can be increased if later appearing peaks are wrongly identified.

        Returns:
            final_df(DataFrame): Dtaframe with found peak positions
            final_df['Found Peaks'] : list of actual peak positions in order of expected peaks
            peaks_data['Other Peaks']: list of unidentified found peaks
        """
        #normalize data
        df, norm_shift = self.normalize_water()
    
        # Split into section with at least 5 columns each
        cols_to_divide = len(df.columns)-1 #-1 because first column is chemical shift
        num_sections = max(cols_to_divide // min_cols_per_section, 1) if cols_to_divide > min_cols_per_section else 1

        cols_per_section = cols_to_divide // num_sections
        extra_cols = cols_to_divide % num_sections  # Extra columns to distribute

        sections = []
        start_col = 1 #ignore first column
        col_sect = []

        for section in range(num_sections):
            extra = 1 if section < extra_cols else 0
            end_col = start_col + cols_per_section + extra
            sections.append(df.iloc[:, [0] + list(range(start_col, end_col))]) #add column with chem_shift to each section
            start_col = end_col
            col_sect.extend([f"{section + 1}"] * (cols_per_section + extra)) #column entries for final dataframe

        reference_peaks = self.expected_peaks
        found_lists = []
        other_lists = []

        #create maximum shift based on assumption, that peaks may shift up to 0.1 during whole experiment. 
        # (1.5 to account for uneven distribution of pH-shifts)
        if max_shift == None:
            max_shift = 0.15 / num_sections 
        
        # Find peaks for each section,max_shift concerning previously found position if exist
        for df_section in sections:
            found, other = self.peak_identify(data_normalized = df_section, reference_peaks = reference_peaks, max_shift= max_shift)
            found_lists.append(found)
            other_lists.append(other)
            reference_peaks = found

        #'unnormalize' the values
        for i in range(len(found_lists)):
            found_lists[i] = [x - norm_shift for x in found_lists[i]]
            other_lists[i] = [x - norm_shift for x in other_lists[i]]

        #create final dataframe
        final_df = pd.DataFrame({
            'Column': df.columns[1:],  # Exclude the chemical shift from the final DataFrame columns
            'section': col_sect
        })

        peaks_data = {'Found Peaks': [], 'Other Peaks': []}

        for i in range(num_sections):
            section_length = len(sections[i].columns) - 1  # -1 to ignore the chemical shift column
            peaks_data['Found Peaks'].extend([found_lists[i]] * section_length)
            peaks_data['Other Peaks'].extend([other_lists[i]] * section_length)


        final_df['Found Peaks'] = peaks_data['Found Peaks']
        final_df['Other Peaks'] = peaks_data['Other Peaks']

        return final_df