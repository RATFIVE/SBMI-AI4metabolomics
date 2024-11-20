# Automated MRI Spectra Metabolite Detector
Analyzing an organism's metabolism is crucial for understanding and monitoring diseases and treatments. Magnetic resonance imaging (MRI) is the only technique to measure the metabolism noninvasively, in vivo, and in vitro.
## Goal 
The goal of this project is to create an application which automatically maps metabolites to recorded yeast spectra, by specifying the reaction metabolite in advance. 

## Plan
16.11 ist Heute
19.11 Meeting:
    - Absprache wie soll das board aussehen(Welceh Plots? Wie sollten die Plots genau assehen(x, achse, y achse usw.))
    - Veränderung über die Zeit
    - Mindestanzahl datenpunkte für ppm finder(Signifikanz?)
    - Deep Learning Model vorstellen.
    - Wie soll die Applikation verpackt sein. Docker, exe, python file?
    - Metadaten. Wie müssen die Aussehen? Nächste Woche -> also nächsten    
    - Wann könnt Ihr unsere App und das fitting 'korrigiert' haben? -> Termin für 1. version

21.11 Übergabe an Tom spätestens und dann implementieren
24.11 Beta Dashboard 

26.11 Meeting: Dashboard Besprechung. Dauer länger
26.11 Start Dokumentation
26.11 Nächste Consulation(9 Uhr) - Deadline Programmieren, Consultatioj

3.12 

10.12

13.12 MoinCC Projektabgabe mit Präsi
19.12 Präsi FH
22.12 Deadline Abgabe Dokument


Fitting Outputs:
**app/**
- **README.md**  
  | Documentation for the application.

- **app.py**  
  | Main script to run the application.

- **.config**  
  | Configuration file for application settings.

- **curve_fitting.py**  
  | Script handling the curve fitting algorithms.

- **DataLoader.py**  
  | Script for loading and processing data.

- **output_dir/**  
  | Directory containing all output results.

  - **File_Name/** (A directory for each processed file)
    |-- **fitted_spectra/**
    |   |-- `file_name_1.csv`
    |   |-- `file_name_2.csv`
    |   |-- ...
    |
    |-- **difference_spectra/**
    |   |-- `file_name_1.csv`
    |   |-- `file_name_2.csv`
    |   |-- ...
    |
    |-- **individual_curves/**
    |   |-- `file_name_1.csv`
    |
    |-- `file_spectra_params.csv`
    |
    |-- `fitted_spectra_params_error.csv`
    |
    |-- `integral_spectra_over_time.csv`
