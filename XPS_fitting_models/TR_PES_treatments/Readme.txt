Python codes to import, pre-process and fit TPRES measurements:
All three codes rely on one Excel file for all inputs: Available_data_summary.xlsx, located in the data folder.
This file has one line per measurement, and associated parameter are along columns.
Only the measurement number (the line) has to be passed to the codes. They will automatically fetch the necessary inputs in the Excel file.

1_Determine_BE_range.py:
- Need only one input: the measurement number
- Necessary for calibration
- Only applied to Pb 4f measurements, having both 4f 7/2 and 4f 5/2 peaks
- Will determine the binding energy resolution of the measurement, based on the fact that Pb 4f 7/2 and 5/2 are separated of 4.8 eV
- Some pre-processing and filtering steps are employed to ensure the detection of both peaks
- The energy resolution obtained with this script is to be inputted in the "inputs table" in the column "Energy resolution (eV)"
- Is only needed once for a given electron detector

2_Import_pre_proces_expe_TRPES.py:
- Need only one input: the measurement number
- Import raw TRPES data files
- Applies three smoothing steps:
	- Aggregation of consecutive spectra in time dimension (this reduces the time resolution)
	- Moving average filter in time dimension
	- Savitzky-Golay filter in energy dimension
- Removes background by employing linear interpolation between both bases of the peaks
- Exports filtered and treated data into .pkl and .csv files

3_Fit_expe_TRPES.py
- Need only one input: the measurement number
- Load pre-processed data from 2_Import_pre_proces_expe_TRPES.py
- Do iterative fits of the spectra
- Gather time evolution of fit parameters
- Export fitted spectra and time evolution of fit parameters into .pkl and .csv files

Fit_expe_functions.py
- File containing functions for 3_Fit_expe_TRPES.py
- Do not try to run this file

Pre_process_functions.py
- File containing functions for 2_Import_pre_proces_expe_TRPES.py
- Do not try to run this file


