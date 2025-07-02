import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from lmfit.models import Model
from lmfit import Parameters
import pickle
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as colours
import sys
import os
from datetime import datetime

# Point to the funcs file
func_path = os.path.abspath('../') + '/Funcs'
sys.path.insert(0, func_path)
from funcs import gaussian, background, flux


# The speed of light
c = 299792.458 #km/s

class Spectrum():
    """
    Generate synthetic spectra and fit these peaks according to the input files.

    Parameters
    ----------
    plotting_info_in : str
        The input filename with peak generating data.
    fitting_info_in : str
        The input filename with peak fitting data.
    lambda_min : int, default=4700
        The minimum wavelength to plot in Angstrom.
    lambda_max : int, default=6800
        The maximum wavelength to plot in Angstrom.
    sig_resolution : float, default=0.5
        The resolution of the hypothetical spectroscopic system.
    sig_sampling : float, default=4.0
        The sampling resolution for fitting the Gaussians.
    bkg : float, default=100
        The background level.
    Nsim : int, default=1000
        The number of simulations to run.
    AoN_min : float, default=0
        The minimum amplitude/nose ratio.
    AoN_max : float, default=10
        The maximum amplitude/noise ratio.

    Methods - Generation
    --------------------
    print_info()
        Print the information about the object.
    get_data()
        Take the input text files and extract the data from them.
    get_line_ratios()
        Get the line ratios for all the peaks.
    create_bkg()
        Create the array for the background level to the spectrum.
    gen_arrs()
        Generate the arrays used to store input and output data.
    generate()
        Generate the synthetic spectrum.
    simulation(plotting=False)
        Generate and fit the synthetic spectrum.
    
    Methods - Analysis
    ------------------
    simulation_independent(plotting=False)
        Generate and fit the synthetic spectrum with all lines treated independently.
    simulation_false(plotting=False)
        Generate and fit the synthetic spectrum, randomly choose some lines to remove.
    output(outfile='peak_data_out.pickle', overwrite=True, raw_data=False)
        Dump out the input and fitted parameters using pickle, can append to data files containing the same number of peaks.
    read_pickle(filename)
        Read data from a pickle file.
    overwrite_all(data_in)
        Overwrite all variables with data from a variable. Designed to correspond to the format of that data is output from this object.
    overwrite(parameter, value)
        Overwrite a particular parameter for plotting the lines with a new value.
    find_relative_error(peak=0, param='sig', ind=None)
        Find the difference between the input and output values of different components.
    find_not_fit(peak=0, param='sig', ind=None)
        Count the number of lines in the data that no fit was found for them based on having a very low standard deviation.
    
    Methods - Plotting
    ------------------
    on_click(event)
        What to do when clicking interactive plots.
    plot_spectrum(y, fit, model, interactive=False)
        Plot a spectrum.
    plot_spectrum_centre(y, fit, model, centre, ran, interactive=False)
        Plot spectra centred on certain wavelengths.
    plot_results(line=0, param='sig', xlim=[-0.2, 11], ylim=[-5, 5], interactive=False, errorbar=False)
        Plot the difference between the input and output values of different components.
    heatmap_sum(param, line, value, brightest=4, text=True, step=1, transparency=False, interactive=False)
        Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the sum of the A/N of all lines.
    heatmap_brightest(param, line, value, brightest=4, text=True, step=1, transparency=False, interactive=False)
        Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line.
    plot_slice(lines, param, bright_l, bright_u, interest_l, interest_u, xlim=[-0.2, 11], ylim=[-5, 5], interactive=False)
        Plot the difference between the input and output values of different components for a certain slice only.
    plot_slice(lines, param, bright_l, bright_u, interest_l, interest_u, xlim=[-0.2, 11], ylim=[-5, 5], interactive=False)
        Plot the difference between the input and output values of different components for a certain slice only.
    scatter_size(param, line, value, brightest=4, step=1, interactive=False)
        Generate scatter plots for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line with the size of the points depending on the number of data points in this range.

    """
    
    def __init__(self, plotting_info_in, fitting_info_in, lambda_min=4700, lambda_max=6800, sig_resolution=0.5, sig_sampling=4.0, bkg=100, Nsim=1000, AoN_min=0, AoN_max=10):     
        self.plotting_info_in = plotting_info_in
        self.fitting_info_in = fitting_info_in
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.sig_resolution = sig_resolution
        self.sig_sampling = sig_sampling
        self.bkg = bkg
        self.Nsim = Nsim
        self.AoN_min = AoN_min
        self.AoN_max = AoN_max
        
        self.peak_params = []
        self.fit_params = []
        self.doublet = []
        self.vel_dep = []
        self.prof_dep = []
        self.peaks_no = 0
        
        # For output
        self.data = np.array([])
        self.data_info = '0  As_in\n1  As_out\n2  As_unc_out\n3  AoNs\n4  AoNs_out\n5  AoNs_unc_out\n6  lams_in\n7  lams_out\n8  lams_unc_out\n9  sig_in\n10 sig_out\n11 sig_unc_out\n12 vels_in\n13 vels_out\n14 vels_unc_out\n15 peak_params\n16 peaks_no\n17 Nsim\n18 doublet\n19 sig_resolution\n20 sig_sampling\n21 This information'
        
        self.get_data()
        
    def print_info(self):
        """
        Print the information about the object.
        
        """
        
        print('Spectrum object, generates and fits synthetic spectra from files.')
        print('----------------')
        print('Input parameters')
        print(f'\t- sig_resolution: {self.sig_resolution}')
        print(f'\t- sig_sampling: {self.sig_sampling}')
        print(f'\t- background level: {self.bkg}')
        print(f'\t- NSim: {self.Nsim}')
        print(f'\t- AoN range: [{self.AoN_min}, {self.AoN_max}')
        print(f'\t- Number of peaks: {self.peaks_no}')

    def get_data(self):
        """
        Take the input text files and extract the data from them.
        
        """
        
        self.peak_params = []
        self.fit_params = []
        self.doublet = []
        self.vel_dep = []
        self.prof_dep = []
        self.peaks_no = 0
        
        # Plotting file
        # plotting_info_in refers to lines_in.txt - initial input spectrum / peaks setup
        with open(self.plotting_info_in) as f:
            data_in = f.readlines()
            
            for i in range(1, len(data_in)):
                entry_in = data_in[i].split()
                
                # Convert data entries from string to float
                param = [float(p) for p in entry_in[2:7]]
                self.peak_params.append(param)

                # Note if a line references itself or a specified line
                if entry_in[7] == 'l':
                    self.doublet.append(i-1)
                elif entry_in[7][0] == 'd':
                    self.doublet.append(int(entry_in[7][1:]))
        
        # How many peaks
        self.peaks_no = len(self.peak_params)

        # Fitting file
        # fitting_info_in refers to fitting.txt - initial input spectrum / peaks setup
        with open(self.fitting_info_in) as f:
            data_fit = f.readlines()
            
            for i in range(1, len(data_fit)):
                entry_fit = data_fit[i].split()
                
                # Ensure that the fitting and plotting lines appear in the same order
                for j in range(1, len(data_in)):
                    entry_in = data_in[j].split()
                    
                    if entry_fit[1] == entry_in[1]:
                        # Convert data entries from string to float
                        param = [float(p) for p in entry_fit[2:7]]
                        self.fit_params.append(param)
        
                        # Note if a line references itself or a specified line
                        if entry_fit[7] == 'l':
                            if entry_fit[8] == 'f':
                                # Completely free
                                self.vel_dep.append(j-1)
                                self.prof_dep.append(j-1)
                                
                            elif entry_fit[8][0] == 't':
                                # Moving with it and same profile
                                self.vel_dep.append(int(entry_fit[8][1:]))
                                self.prof_dep.append(int(entry_fit[8][1:]))
                                
                            elif entry_fit[8][0] == 'v':
                                # Multiple species moving together but profiles different
                                self.vel_dep.append(int(entry_fit[8][1:]))
                                self.prof_dep.append(i)
                                
                        elif entry_fit[7][0] == 'd':
                            # If dependent then moving with it and has same profile
                            self.vel_dep.append(int(entry_fit[7][1:]))
                            self.prof_dep.append(int(entry_fit[7][1:]))
        
        # Give the lines the same velocity and sigma as the lines they are dependent on, i.e. ignore velocity and sigma inputs for lines that are dependent on others
        for i in range(self.peaks_no - 1):
            self.peak_params[i][3] = self.peak_params[self.doublet[i]][3]
            self.peak_params[i][4] = self.peak_params[self.doublet[i]][4]
            self.vel_dep[i] = self.vel_dep[self.doublet[i]]
            self.prof_dep[i] = self.prof_dep[self.doublet[i]]
            
            self.fit_params[i][3] = self.fit_params[self.vel_dep[i]][3]
            self.fit_params[i][4] = self.fit_params[self.prof_dep[i]][4]
        
        self.get_line_ratios()
    
    def get_line_ratios(self):
        """
        Get the line ratios for all the peaks.

        """
        
        # Get the line ratios
        self.line_ratios = np.array(self.peak_params)[:,1]
        for i in range(self.peaks_no):
            if self.doublet[i] != i:
                self.line_ratios[i] = self.line_ratios[i] * self.line_ratios[self.doublet[i]]

    def create_bkg(self):
        """
        Create the array for the background level to the spectrum.
        
        """
        
        # dx = self.sig_resolution / self.sig_sampling
        # nx = int(self.lambda_max - self.lambda_min)
        # self.x = np.linspace(self.lambda_min, self.lambda_max, nx)

        lam_min = min(p[0] * (1 + p[3]/c) for p in self.peak_params)
        lam_max = max(p[0] * (1 + p[3]/c) for p in self.peak_params)
        sig_in = max(p[4] / c * p[0] * (1 + p[4]/c) for p in self.peak_params)
        dx = self.sig_resolution / self.sig_sampling
        nx = int(2 * (20*sig_in/dx) + 1)
        self.x = np.linspace(-20 * sig_in + lam_min, 20 * sig_in + lam_max, nx)
   
    def gen_arrs(self):
        """
        Generate the arrays used to store input and output data.
   
        """
        
        # Initialize input arrays
        self.As_in = np.empty((self.peaks_no, self.Nsim))
        self.sig_in = np.empty((self.peaks_no, self.Nsim))
        self.vels_in = np.empty((self.peaks_no, self.Nsim))
        self.lams_in = np.array([p[0] for p in self.peak_params])

        # Initialize output arrays
        self.As_out = np.empty((self.peaks_no, self.Nsim))
        self.As_unc_out = np.empty((self.peaks_no, self.Nsim))
        self.sig_out = np.empty((self.peaks_no, self.Nsim))
        self.sig_unc_out = np.empty((self.peaks_no, self.Nsim))
        self.vels_out = np.empty((self.peaks_no, self.Nsim))
        self.vels_unc_out = np.empty((self.peaks_no, self.Nsim))
        self.lams_out = np.empty((self.peaks_no, self.Nsim))
        self.lams_unc_out = np.empty((self.peaks_no, self.Nsim))
        self.AoNs_out = np.empty((self.peaks_no, self.Nsim))
        self.AoNs_unc_out = np.empty((self.peaks_no, self.Nsim))
        
        # Store data and fit
        self.spectra_mat = np.empty((self.Nsim, len(self.x)))
        self.model_mat = np.empty((self.Nsim, len(self.x)))
        self.fit_mat = np.empty((self.Nsim, len(self.x)))
    
    def init_model(self):
        """
        Generates the background level of the model.
   
        """
        
        # Prerequesite functions
        self.create_bkg()
        self.gen_arrs()

        self.model = background(self.x, self.bkg)
        gaussian_models = []
        self.mod = Model(background, prefix='bkg_')

        # Loop through for the number of peaks
        for i, (lam, relative_A_l, relative_A_u, vel, sig) in enumerate(self.peak_params):
            gauss = Model(gaussian, prefix=f'g{i}_')
            self.mod += gauss
            gaussian_models.append(gauss)

    def generate(self):
        """
        Generate the synthetic spectrum.

        Returns
        -------
        y_vals : array
            The generated spectra.
    
        """
        
        # Initialise
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        
        y_vals = np.empty((self.Nsim, len(self.x)))

        for index, AoN in enumerate(self.AoNs):
            A = np.sqrt(self.bkg) * AoN

            # Generate Gaussian + Noise data 
            self.model = background(self.x, self.bkg)
            
            # Generate free and doublet lines separately
            amplitudes = []
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if self.doublet[i] == i:
                    # If given a range of values choose a random value
                    relative_A = np.random.uniform(relative_A_l, relative_A_u)
                    amplitudes.append(relative_A)
                    self.model += gaussian(self.x, A * relative_A, lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A
                    self.sig_in[i][index] = sig
                    self.vels_in[i][index] = vel
                else:
                    amplitudes.append(np.nan)    
            
            # Repeat to generate the doublet lines
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if np.isnan(amplitudes[i]):
                    # If given a range of values choose a random value
                    relative_A = np.random.uniform(relative_A_l, relative_A_u)
                    self.model += gaussian(self.x, A * relative_A * amplitudes[self.doublet[i]], lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A * amplitudes[self.doublet[i]]
                    self.sig_in[i][index] = sig
                    self.vels_in[i][index] = vel
                
            # Generate noise and add it to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            y_vals[index] = y
        
        return y_vals
    
    def simulation(self, plotting=False):
        """
        Generate and fit the synthetic spectrum.
        
        Parameters
        ----------
        plotting : bool, default=False
            Whether or not to plot the graphs.
            
        See Also
        --------
        simulation_independent : Generate and fit the synthetic spectrum with all lines treated independently.
        simulation_false : Generate and fit the synthetic spectrum, randomly choose some lines to remove.
        
        """
     
        # Initialise
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        for index, AoN in enumerate(self.AoNs):
            A = np.sqrt(self.bkg) * AoN
            
            # Generate Gaussian + Noise data 
            self.model = background(self.x, self.bkg)
            
            # Generate free and doublet lines separately
            amplitudes = []
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if self.doublet[i] == i:
                    # If given a range of values choose a random value
                    relative_A = np.random.uniform(relative_A_l, relative_A_u)
                    amplitudes.append(relative_A)
                    self.model += gaussian(self.x, A * relative_A, lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A
                    self.sig_in[i][index] = sig
                    self.vels_in[i][index] = vel
                else:
                    amplitudes.append(np.nan)    
            
            # Repeat to generate the doublet lines
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if np.isnan(amplitudes[i]):
                    # If given a range of values choose a random value
                    relative_A = np.random.uniform(relative_A_l, relative_A_u)
                    self.model += gaussian(self.x, A * relative_A * amplitudes[self.doublet[i]], lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A * amplitudes[self.doublet[i]]
                    self.sig_in[i][index] = sig
                    self.vels_in[i][index] = vel
                
            # Generate noise and add it to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            
            # Fit with LMfit 
            pfit = Parameters()
        
            # The background level
            pfit.add('bkg_bkg', value=self.bkg, vary=True)
            
            # Setting up parameters for the peaks (g_i)
            for i in range(self.peaks_no):
                # These values are fixed either physically or by the instrument
                pfit.add(f'g{i}_lam_rf', value=self.fit_params[i][0], vary=False)
                pfit.add(name=f'g{i}_sig_resolution', value=self.sig_resolution, vary=False)
                
                if self.doublet[i] == i:
                    # For free lines take initial guess as largest y value in the region +- 100 Angstrom from where it should be based on initial guesses
                    expec_lam = self.fit_params[i][0] *  (1 + self.fit_params[i][3]/c)
                    ind = np.where(np.abs(expec_lam - self.x) <= 100)
                    pfit.add(f'g{i}_A', value=np.max(y[ind]) - np.median(y), min=self.fit_params[i][1], max=self.fit_params[i][2])
                    
                    # If independent in terms of velocity and sigma take those as its initial estimates
                    if self.vel_dep[i] == i:
                        pfit.add(f'g{i}_vel', value=self.fit_params[i][3])
                    if self.prof_dep[i] == i:
                        pfit.add(f'g{i}_sig', value=self.fit_params[i][4])

            # Loop again for amplitudes, velocities and sigmas of dependent peaks - have to do afterwards as could be dependent on a line appearing after it in the file
            for i in range(self.peaks_no):
                if self.vel_dep[i] != i:
                    # Velocity must be that of another peak
                    pfit.add(f'g{i}_vel', expr=f'g{self.vel_dep[i]}_vel')
                if self.prof_dep[i] != i:
                    # Sigma must be that of another peak
                    pfit.add(f'g{i}_sig', expr=f'g{self.prof_dep[i]}_sig')
                if self.doublet[i] != i:
                    # The amplitude of the peak is equal to the that of the reference line times some value
                    if self.fit_params[i][1] != self.fit_params[i][2]:
                        pfit.add(f'g{i}_delta', min=self.fit_params[i][1], max=self.fit_params[i][2])
                    else:
                        pfit.add(f'g{i}_delta', expr=f'{self.fit_params[i][2]}')
                    pfit.add(f'g{i}_A', expr=f'g{i}_delta * g{self.doublet[i]}_A')
                
                
            fit = self.mod.fit(y, pfit, x=self.x)
            
            # Save generated data
            for peak in range(self.peaks_no):
                self.As_out[peak][index] = fit.params[f'g{peak}_A'].value
                self.As_unc_out[peak][index] = fit.params[f'g{peak}_A'].stderr
                self.sig_out[peak][index] = fit.params[f'g{peak}_sig'].value
                self.sig_unc_out[peak][index] = fit.params[f'g{peak}_sig'].stderr
                self.vels_out[peak][index] = fit.params[f'g{peak}_vel'].value
                self.vels_unc_out[peak][index] = fit.params[f'g{peak}_vel'].stderr
                self.lams_out[peak][index] = fit.params[f'g{peak}_lam_rf'].value
                self.lams_unc_out[peak][index] = fit.params[f'g{peak}_lam_rf'].stderr
            
            self.spectra_mat[index] = y
            self.model_mat[index] = self.model
            self.fit_mat[index] = fit.best_fit
                

            # Plotting
            if plotting == True:
                self.plot_spectrum(y, fit.best_fit, self.model)
        
        self.f_out, self.f_unc_out = flux(self.As_out, self.sig_out, self.As_unc_out, self.sig_unc_out)
        self.f_in, self.f_unc_in = flux(self.As_in, self.sig_in)
        
        # Save AoNs
        for peak in range(self.peaks_no):
            self.AoNs_out[peak] = self.As_out[peak]/np.sqrt(self.bkg)
            self.AoNs_unc_out[peak] = self.As_unc_out[peak]/np.sqrt(self.bkg)

# All methods above this point are necessary for the generation and fitting of the spectra
# Below this point the methods are useful for testing other things and extracting and observing the data

    def simulation_independent(self, plotting=False):
        """
        Generate and fit the synthetic spectrum with all lines treated independently.
        
        Parameters
        ----------
        plotting : bool, default=False
            Whether or not to plot the graphs.

        See Also
        --------
        simulation : Generate and fit the synthetic spectrum.   
        simulation_false : Generate and fit the synthetic spectrum, randomly choose some lines to remove.
        
        """
     
        # Initialise
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        for index, AoN in enumerate(self.AoNs):
            A = np.sqrt(self.bkg) * AoN
            
            # Generate Gaussian + Noise data 
            self.model = background(self.x, self.bkg)
            
            # Generate free and doublet lines separately
            amplitudes = []
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                # If given a range of values choose a random value
                relative_A = np.random.uniform(relative_A_l, relative_A_u)
                amplitudes.append(relative_A)
                self.model += gaussian(self.x, A * relative_A, lam, vel, sig, self.sig_resolution)
                
                # Store input data
                self.As_in[i][index] = A * relative_A
                self.sig_in[i][index] = sig
                self.vels_in[i][index] = vel
                
            # Generate noise and add it to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            
            # Fit with LMfit 
            pfit = Parameters()
        
            # The background level
            pfit.add('bkg_bkg', value=self.bkg, vary=True)
            
            # Setting up parameters for the peaks (g_i)
            for i in range(self.peaks_no):
                # These values are fixed either physically or by the instrument
                pfit.add(f'g{i}_lam_rf', value=self.fit_params[i][0], vary=False)
                pfit.add(name=f'g{i}_sig_resolution', value=self.sig_resolution, vary=False)
                
                # Take initial guess as largest y value in the region +- 100 Angstrom from where it should be based on initial guesses
                expec_lam = self.fit_params[i][0] *  (1 + self.fit_params[i][3]/c)
                ind = np.where(np.abs(expec_lam - self.x) <= 100)
                pfit.add(f'g{i}_A', value=np.max(y[ind]) - np.median(y), min=0, max=np.inf)
                pfit.add(f'g{i}_vel', value=self.fit_params[i][3])
                pfit.add(f'g{i}_sig', value=self.fit_params[i][4])
                

              
            fit = self.mod.fit(y, pfit, x=self.x)
            
            # Save generated data
            for peak in range(self.peaks_no):
                self.As_out[peak][index] = fit.params[f'g{peak}_A'].value
                self.As_unc_out[peak][index] = fit.params[f'g{peak}_A'].stderr
                self.sig_out[peak][index] = fit.params[f'g{peak}_sig'].value
                self.sig_unc_out[peak][index] = fit.params[f'g{peak}_sig'].stderr
                self.vels_out[peak][index] = fit.params[f'g{peak}_vel'].value
                self.vels_unc_out[peak][index] = fit.params[f'g{peak}_vel'].stderr
                self.lams_out[peak][index] = fit.params[f'g{peak}_lam_rf'].value
                self.lams_unc_out[peak][index] = fit.params[f'g{peak}_lam_rf'].stderr
            
            self.spectra_mat[index] = y
            self.model_mat[index] = self.model
            self.fit_mat[index] = fit.best_fit
                

            # Plotting
            if plotting == True:
                self.plot_spectrum(y, fit.best_fit, self.model)
        
        self.f_out, self.f_unc_out = flux(self.As_out, self.sig_out, self.As_unc_out, self.sig_unc_out)
        self.f_in, self.f_unc_in = flux(self.As_in, self.sig_in)
        
        # Save AoNs
        for peak in range(self.peaks_no):
            self.AoNs_out[peak] = self.As_out[peak]/np.sqrt(self.bkg)
            self.AoNs_unc_out[peak] = self.As_unc_out[peak]/np.sqrt(self.bkg)

    def simulation_false(self, plotting=False):
        """
        Generate and fit the synthetic spectrum, randomly choose some lines to remove.
        
        Parameters
        ----------
        plotting : bool, default=False
            Whether or not to plot the graphs.
            
        See Also
        --------
        simulation : Generate and fit the synthetic spectrum.
        simulation_independent : Generate and fit the synthetic spectrum with all lines treated independently.
        
        """
        
        # Initial variables
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        
        # Keep some lines, remove others
        self.keep_lines = np.random.choice([0,1], (self.peaks_no, self.Nsim))
    
        # Generate and fit spectra
        for index, AoN in enumerate(self.AoNs):
            A = np.sqrt(self.bkg) * AoN
            
            # Generate Gaussian + Noise data
            self.model = background(self.x, self.bkg)
            
            # Generate free and doublet lines separately
            amplitudes = []
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if self.doublet[i] == i:
                    # If given a range of values choose a random value
                    # Randomly keep or remove a line
                    relative_A = np.random.uniform(relative_A_l, relative_A_u) * self.keep_lines[i][index]
                    amplitudes.append(relative_A)
                    self.model += gaussian(self.x, A * relative_A, lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A
                    self.sig_in[i][index] = sig
                    self.vels_in[i][index] = vel
                else:
                    amplitudes.append(np.nan)    
            
            # Repeat to generate the doublet lines
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if np.isnan(amplitudes[i]):
                    
                    # If a main line is removed the doublet should also be removed
                    if self.keep_lines[self.doublet[i]][index] == 0:
                        self.keep_lines[i][index] = 0    
                    
                    # If given a range of values choose a random value
                    # Randomly keep or remove a line
                    relative_A = np.random.uniform(relative_A_l, relative_A_u) * self.keep_lines[i][index]
                    self.model += gaussian(self.x, A * relative_A * amplitudes[self.doublet[i]], lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A * amplitudes[self.doublet[i]]
                    self.sig_in[i][index] = sig
                    self.vels_in[i][index] = vel
                
            # Generate noise and add to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            
            # Fit with LMfit 
            pfit = Parameters()
        
            # The background level
            pfit.add('bkg_bkg', value=self.bkg, vary=True)
            
            # Setting up parameters for the peaks (g_i)
            for i in range(self.peaks_no):
                # These values are fixed either physically or by the instrument
                pfit.add(f'g{i}_lam_rf', value=self.fit_params[i][0], vary=False)
                pfit.add(name=f'g{i}_sig_resolution', value=self.sig_resolution, vary=False)
                
                if self.doublet[i] == i:
                    # For free lines take initial guess as largest y value in the region +- 100 Angstrom from where it should be based on initial guesses
                    expec_lam = self.fit_params[i][0] *  (1 + self.fit_params[i][3]/c)
                    ind = np.where(np.abs(expec_lam - self.x) <= 100)
                    pfit.add(f'g{i}_A', value=np.max(y[ind]) - np.median(y), min=self.fit_params[i][1], max=self.fit_params[i][2])
                    
                    # If independent in terms of velocity and sigma take those as its initial estimates
                    if self.vel_dep[i] == i:
                        pfit.add(f'g{i}_vel', value=self.fit_params[i][3])
                    if self.prof_dep[i] == i:
                        pfit.add(f'g{i}_sig', value=self.fit_params[i][4])

            # Loop again for amplitudes, velocities and sigmas of dependent peaks - have to do afterwards as could be dependent on a line appearing after it in the file
            for i in range(self.peaks_no):
                if self.vel_dep[i] != i:
                    # Velocity must be that of another peak
                    pfit.add(f'g{i}_vel', expr=f'g{self.vel_dep[i]}_vel')
                if self.prof_dep[i] != i:
                    # Sigma must be that of another peak
                    pfit.add(f'g{i}_sig', expr=f'g{self.prof_dep[i]}_sig')
                if self.doublet[i] != i:
                    # The amplitude of the peak is equal to the that of the reference line times some value
                    if self.fit_params[i][1] != self.fit_params[i][2]:
                        pfit.add(f'g{i}_delta', min=self.fit_params[i][1], max=self.fit_params[i][2])
                    else:
                        pfit.add(f'g{i}_delta', expr=f'{self.fit_params[i][2]}')
                    pfit.add(f'g{i}_A', expr=f'g{i}_delta * g{self.doublet[i]}_A')
                
                
            fit = self.mod.fit(y, pfit, x=self.x)
            
            # Save generated data
            for peak in range(self.peaks_no):
                self.As_out[peak][index] = fit.params[f'g{peak}_A'].value
                self.As_unc_out[peak][index] = fit.params[f'g{peak}_A'].stderr
                self.sig_out[peak][index] = fit.params[f'g{peak}_sig'].value
                self.sig_unc_out[peak][index] = fit.params[f'g{peak}_sig'].stderr
                self.vels_out[peak][index] = fit.params[f'g{peak}_vel'].value
                self.vels_unc_out[peak][index] = fit.params[f'g{peak}_vel'].stderr
                self.lams_out[peak][index] = fit.params[f'g{peak}_lam_rf'].value
                self.lams_unc_out[peak][index] = fit.params[f'g{peak}_lam_rf'].stderr
                self.f_out, self.f_unc_out = flux(self.As_out, self.sig_out, self.As_unc_out, self.sig_unc_out)
                self.f_in, self.f_unc_in = flux(self.As_in, self.sig_in)

            self.spectra_mat[index] = y
            self.model_mat[index] = self.model
            self.fit_mat[index] = fit.best_fit

            # Plotting
            if plotting == True:
                self.plot_spectrum(y, fit.best_fit, self.model)
        
        # Save AoNs
        for peak in range(self.peaks_no):
            self.AoNs_out[peak] = self.As_out[peak]/np.sqrt(self.bkg)
            self.AoNs_unc_out[peak] = self.As_unc_out[peak]/np.sqrt(self.bkg)

    def output(self, outfile='peak_data_out.pickle', overwrite=True, raw_data=False):
        """
        Dump out the input and fitted parameters using pickle, can append to data files containing the same number of peaks.

        Parameters
        ----------
        outfile : str, default='peak_data_out.pickle'
            The name of the file to save the data to.
        overwrite : bool, default=True
            Overwrite the file or append to it.
        raw_data : bool, deafult=False
            Return the entire spectra, model and fit.
           
        """
        
        data = [self.As_in, self.As_out, self.As_unc_out, self.AoNs, self.AoNs_out, self.AoNs_unc_out, self.lams_in, self.lams_out, self.lams_unc_out, self.sig_in, self.sig_out, self.sig_unc_out, self.vels_in, self.vels_out, self.vels_unc_out, self.peak_params, self.peaks_no, self.Nsim, self.doublet, self.sig_resolution, self.sig_sampling]
        
        if raw_data == True:
            data.append(self.spectra_mat)
            data.append(self.model_mat)
            data.append(self.fit_mat)
            
            self.data_info = '0  As_in\n1  As_out\n2  As_unc_out\n3  AoNs\n4  AoNs_out\n5  AoNs_unc_out\n6  lams_in\n7  lams_out\n8  lams_unc_out\n9  sig_in\n10 sig_out\n11 sig_unc_out\n12 vels_in\n13 vels_out\n14 vels_unc_out\n15 peak_params\n16 peaks_no\n17 Nsim\n18 doublet\n19 sig_resolution\n20 sig_sampling\n21 spectra_mat\n22 model_mat\n23 fit_mat\n24 This information'
        
        # Don't overwrite an already existing file unless desired
        if overwrite == False:
            try:
                # Try to open the data file if it exists and append the data from this run to it, increasing Nsim as needed, if they are compatible
                with open(outfile, 'rb') as pickle_file:
                    in_data = pickle.load(pickle_file)
                 
                # Check that same resolutions used in both
                if in_data[19] == self.sig_resolution and in_data[20] == self.sig_sampling:
                 
                    # Concatenate the data
                    for i in range(len(data)):
                        if i != 6 and i < 15:
                            data[i] = np.concatenate((in_data[i].T, data[i].T)).T
                        
                        if i >= 21 and i < 24:
                            data[i] = np.concatenate((in_data[i], data[i]))
                    
                    # Increment Nsim
                    data[17] = data[17] + in_data[17]
                    
                    # Add the info
                    data.append(self.data_info)
                    
                    # Save the data
                    with open(outfile, 'wb') as outfile:
                        pickle.dump(data, outfile)
                    outfile.close()
                
                else:
                    # If the data are not compatible then make a new file named based on the current timestamp
                    dt = datetime.now().strftime("%Y%m%dT%H%M%S")
                    
                    print('File not updated: The same sampling and resolutions must be used throughout the file.\nData saved to {dt}-{outfile} instead')

                    outfile = open(f'{dt}-{outfile}', 'wb')
                    data.append(self.data_info)
                    
                    # Write the data
                    pickle.dump(data, outfile)
                    outfile.close()
                
            except:
                # Create the file
                outfile = open(outfile, 'wb')
                
                data.append(self.data_info)
                
                # Write the data
                pickle.dump(data, outfile)
                outfile.close()
        else:
            # Create the file
            outfile = open(outfile, 'wb')
            
            data.append(self.data_info)
        
            # Write the data
            pickle.dump(data, outfile)
            outfile.close()
    
    def read_pickle(self, filename):
        """
        Read data from a pickle file.

        Parameters
        ----------
        filename : str
            The filename to read.
    
        """
      
        with open(filename, 'rb') as pickle_file:
            self.pickle_in = pickle.load(pickle_file)
    
    def overwrite_all(self, data_in):
        """
        Overwrite all variables with data from a variable. Designed to correspond to the format of that data is output from this object.

        Parameters
        ----------
        data_in : array
            The data with which to overwrite the current variables.
        
        See Also
        --------
        overwrite : Overwrite a particular parameter for plotting the lines with a new value.
     
        """
        
        # Overwrite variables
        self.As_in = data_in[0]
        self.As_out = data_in[1]
        self.As_unc_out = data_in[2]
        self.AoNs = data_in[3]
        self.AoNs_out = data_in[4]
        self.AoNs_unc_out = data_in[5]
        self.lams_in = data_in[6]
        self.lams_out = data_in[7]
        self.lams_unc_out = data_in[8]
        self.sig_in = data_in[9]
        self.sig_out = data_in[10]
        self.sig_unc_out = data_in[11]
        self.vels_in = data_in[12]
        self.vels_out = data_in[13]
        self.vels_unc_out = data_in[14]
        self.peak_params = data_in[15]
        self.peaks_no = data_in[16]
        self.Nsim = data_in[17]
        self.doublet = data_in[18]
        self.sig_resolution = data_in[19]
        self.sig_sampling = data_in[20]
        
        if len(data_in) == 25:
            self.spectra_mat = data_in[21]
            self.model_mat = data_in[22]
            self.fit_mat = data_in[23]
        
        # Calculate fluxes
        self.f_out, self.f_unc_out = flux(self.As_out, self.sig_out, self.As_unc_out, self.sig_unc_out)
        self.f_in, self.f_unc_in = flux(self.As_in, self.sig_in)
        
        # Get ratios
        self.get_line_ratios()
        
        # Get x data
        self.create_bkg()

    def overwrite(self, parameter, value):
        """
        Overwrite a particular parameter for plotting the lines with a new value.

        Parameters
        ----------
        parameter : int
            The parameter to overwrite given as the index of the parameter in self.peak_params.
        value
            The (list of) values to overwrite with.
        
        Indices
        -------
        wl,	A_in_l,	A_in_u,	v_in,	sig_in,	free
        0,  1,      2,      3,     4,      5
        
        See Also
        --------
        overwrite_all : Overwrite all variables with data from a variable. Designed to correspond to the format of that data is output from this object.
        
        """

        for i in range(self.peaks_no):
            if type(value) == list:
                self.peak_params[i][parameter] = value[i]
                self.fit_params[i][parameter] = value[i]
            else:
                self.peak_params[i][parameter] = value
                self.fit_params[i][parameter] = value

    def find_relative_error(self, peak=0, param='sig', ind=None):
        """
        Find the difference between the input and output values of different components.
        
        Parameters
        ----------
        peak : int, default=0
            Which peak of the spectrum to check for.
        param : {'sig', 'vel', 'A', 'flux'}, default='sig'
            Which parameter to check for.
        ind : array, defualt=None
            Which specific elements to check, if None checks all.
            
        Returns
        -------
        arr : array
            The values of difference between the input and output values of the selected component.
        std : float
            The standard deviation of arr.
        median : float
            The median of arr.
   
        """
        
        if param == 'sig':
            array = (self.sig_out - self.sig_in) / self.sig_in
        elif param == 'vel':
            array = (self.vels_out - self.vels_in) / self.vels_in
        elif param == 'A':
            array = (self.As_out - self.As_in) / self.As_in
        elif param == 'flux':
            array = (self.f_out - self.f_in) / self.f_in

        arr = array[peak]
        if ind is not None:
            arr = arr[ind][0]
        std = np.std(arr)
        med = np.median(arr)
        
        return arr, std, med

    def find_not_fit(self, peak=0, param='sig', ind=None):
        """
        Count the number of lines in the data that no fit was found for them based on having a very low standard deviation.
        
        Parameters
        ----------
        peak : int, default=0
            Which peak of the spectrum to check for.
        param : {'sig', 'vel', 'A', 'flux'}, default='sig'
            Which parameter to check for.
        ind : array, defualt=None
            Which specific elements to check, if None checks all.
        
        Returns
        -------
        close_0 : array
            The indices where sig_out is close to 0.
        """
        
        # Find where param is close to 0
        
        close_0 = np.unique(np.where(np.abs(self.find_relative_error(peak=peak, param=param, ind=ind)[0] + 1) <= 0.01)[0])
        
        return close_0

# Below this point all methods have something to do with plotting the data

    def on_click(self, event):
        """
        What to do when clicking interactive plots.
        
        """
                
        x_click, y_click = event.xdata, event.ydata
        
        if current_plot == 'heatmap' or current_plot == 'scatter size' or current_plot == 'heatmap sum':
            # Upper and lower x and y limits
            global bright_l, bright_u, interest_l, interest_u
            bright_l = (x_click//step_of_interest) * step_of_interest
            bright_u = bright_l + step_of_interest
            interest_l = (y_click//step_of_interest) * step_of_interest
            interest_u = interest_l + step_of_interest
        else:
            closest_point_ind = np.argmin(np.sqrt((self.AoNs_out[line_of_interest] - x_click)**2 + (arr_of_interest[line_of_interest] - y_click)**2))
        
        # Just scatter plots
        # Scatter to spectrum
        if event.button == 1 and current_plot == 'results' and event.dblclick == True:
            plt.close()
            self.plot_spectrum_centre(self.spectra_mat[closest_point_ind], self.fit_mat[closest_point_ind], self.model_mat[closest_point_ind], [4950, 6650], 150, interactive=True)
        # Spectrum back to scatter
        elif event.button == 3 and current_plot == 'spectrum' and event.dblclick == True and heatmap == False and scat_size == False and heatmap_sum == False:
            plt.close()
            self.plot_results(line=line_of_interest, param=param_of_interest, xlim=xlim_of_interest, ylim=ylim_of_interest, interactive=True)
        # Heatmaps and scatter size plots
        # Heatmaps/scatter size to scatter or spectrum to scatter
        elif ((event.button == 1 and (current_plot == 'heatmap' or current_plot == 'scatter size' or current_plot == 'heatmap sum')) or (event.button == 3 and current_plot == 'spectrum')) and event.dblclick == True:
            plt.close()
            print(self.no_points)
            print(arr_of_interest)
            self.plot_slice([line_of_interest, brightest_of_interest], param_of_interest, bright_l, bright_u, interest_l, interest_u, interactive=True)
        # Slice to heatmap
        elif event.button == 3 and current_plot == 'slice' and heatmap == True and event.dblclick == True:
            plt.close()
            self.heatmap_brightest(param_of_interest, line_of_interest, value_of_interest, brightest_of_interest, show_text, step_of_interest, transparent, interactive=True)
        # Slice to scatter size
        elif event.button == 3 and current_plot == 'slice' and scat_size == True and event.dblclick == True:
            plt.close()
            self.scatter_size(param_of_interest, line_of_interest, value_of_interest, brightest_of_interest, step_of_interest, interactive=True)
        # Slice to heatmap sum
        elif event.button == 3 and current_plot == 'slice' and heatmap_sum == True and event.dblclick == True:
            plt.close()
            self.heatmap_sum(param_of_interest, line_of_interest, value_of_interest, brightest_of_interest, show_text, step_of_interest, transparent, interactive=True)
        # Slice to spectrum
        elif event.button == 1 and current_plot == 'slice' and event.dblclick == True:            
            if event.inaxes in [axis[0]]:
                closest_point_ind = np.argmin(np.sqrt((self.AoNs_out[line_of_interest] - x_click)**2 + (arr_of_interest[line_of_interest] - y_click)**2))
            if event.inaxes in [axis[1]]:
                closest_point_ind = np.argmin(np.sqrt((self.AoNs_out[brightest_of_interest] - x_click)**2 + (arr_of_interest[brightest_of_interest] - y_click)**2))
            plt.close()
            self.plot_spectrum_centre(self.spectra_mat[closest_point_ind], self.fit_mat[closest_point_ind], self.model_mat[closest_point_ind], [4950, 6650], 150, interactive=True)
            pass

    def plot_spectrum(self, y, fit, model, interactive=False):
        """
        Plot a spectrum.

        Parameters
        ----------
        y : array
            The amplitude data.
        fit : array
            The fit data for the model.
        model : array
            The model data.
        interactive : bool, default=False
            Whether to have it be interactive or not.

        See Also
        --------
        plot_spectrum_centre : Plot spectra centred on certain wavelengths.
      
        """
        
        fig, ax = plt.subplots()

        labels = ['input model + noise', 'input model', 'fitted model']
        plt.plot(self.x, y, 'k-')
        plt.plot(self.x, model, 'c-')
        plt.plot(self.x, fit, 'r-')
        plt.xlabel(r'$\lambda$ ($\AA$)')
        plt.ylabel('Amplitude (arbitrary units)')
        # plt.title('Generated and fit spectrum with emission lines')
        plt.title('One of the spectra used in this project')
        plt.legend(labels)
        plt.grid()
        plt.show()
        
        if interactive == True:
            fig.canvas.mpl_connect('button_press_event', self.on_click)    

    def plot_spectrum_centre(self, y, fit, model, centre, ran, interactive=False):
        """
        Plot spectra centred on certain wavelengths.

        Parameters
        ----------
        y : array
            The amplitude data.
        fit : array
            The fit data for the model.
        model : array
            The model data.
        centre : array
            The wavelengths to centre the plots on.
        ran : float
            The range to plot from, centre - ran to centre + ran.
        interactive : bool, default=False
            Whether to have it be interactive or not.
        
        See Also
        --------
        plot_spectrum : Plot a spectrum.
        
        """
        
        fig, ax = plt.subplots(1, len(centre))

        labels = ['input model + noise', 'input model', 'fitted model']
        for i in range(len(centre)):
            # Which areas to plot
            # ind = np.where(np.abs(self.x - centre[i]) <= ran)
            
            ax[i].plot(self.x, y, 'k-')
            ax[i].plot(self.x, model, 'c-')
            ax[i].plot(self.x, fit, 'r-')
            ax[i].set_xlabel(r'$\lambda$ ($\AA$)')
            # ax[i].axhline
            ax[i].grid()
            ax[i].set_xlim([centre[i] - ran, centre[i] + ran])
            
        ax[0].set_ylabel('Amplitude (arbitrary units)')
        ax[0].legend(labels)
        fig.suptitle('Generated and fit spectrum with emission lines')
        plt.show()
        
        if interactive == True:
            global current_plot
            current_plot = 'spectrum'
            
            fig.canvas.mpl_connect('button_press_event', self.on_click)

    def plot_results(self, line=0, param='sig', xlim=[-0.2, 11], ylim=[-5, 5], interactive=False, errorbar=False):
        """
        Plot the difference between the input and output values of different components.
        
        Parameters
        ----------
        line : int, default=0
            Which line of the spectrum to plot for.
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to plot for. 
        xlim : array or None, default=[-0.2,11]
            The xlimits of the plot.
        ylim : array or None, default=[-5,5]
            The ylimits of the plot.
        interactive : bool, default=False
            Make interactive plots or not.
        errorbar : bool, default=False
            Whether to plot the errorbars.
     
        """
 
        if param == 'sig':
            array = (self.sig_out - self.sig_in) / self.sig_in
            unc = self.sig_unc_out / self.sig_in
        elif param == 'vel':
            array = (self.vels_out - self.vels_in) / self.vels_in
            unc = self.vels_unc_out / self.vels_in
        elif param == 'A':
            array = (self.As_out - self.As_in) / self.As_in
            unc = self.As_unc_out / self.As_in
        elif param == 'flux':
            array = (self.f_out - self.f_in) / self.f_in
            unc = self.f_unc_out / self.f_in
        
        close_0 = self.find_not_fit(peak=line, param=param)
        
        fig, ax = plt.subplots()
        
        label = f'({param}_out - {param}_in)/{param}_in'
        plt.title(rf'{label} against A/N of peak {line} for Nsim = {self.Nsim}'+f'\nv_in = {self.peak_params[0][3]}, sig_in = {self.peak_params[0][4]}')
        plt.axhline(0, color='lightgrey')
        plt.scatter(self.AoNs_out[line], array[line], s=2, label='Data', zorder=2.5)
        plt.scatter(self.AoNs_out[line][close_0], array[line][close_0], s=2, zorder=2.5)
        if errorbar == True:
            plt.errorbar(self.AoNs_out[line], array[line], unc[line], fmt='none', zorder=2.5)
            plt.errorbar(self.AoNs_out[line][close_0], array[line][close_0], unc[line][close_0], fmt='none', color='#ff7f0e', zorder=2.5)
        plt.xlabel('A/N')
        plt.ylabel(label)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.show()
        if interactive == True:
            global line_of_interest, arr_of_interest, param_of_interest, xlim_of_interest, ylim_of_interest
            line_of_interest = line
            arr_of_interest = array
            param_of_interest = param
            xlim_of_interest = xlim
            ylim_of_interest = ylim
            
            global current_plot, heatmap, scat_size, heatmap_sum
            current_plot = 'results'
            heatmap = False
            scat_size = False
            heatmap_sum = False
            
            fig.canvas.mpl_connect('button_press_event', self.on_click)
                
    def heatmap_sum(self, param, line, value, brightest=4, text=True, step=1, transparency=False, interactive=False):
        """
        Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the sum of the A/N of all lines.

        Parameterss
        ----------
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to check for.
        line : int
            The line of interest.
        value : {'std', 'median'}
            Plot the standard deviations or the medians.
        brightest : int, default=4
            The brightest line (default corrseponds to H alpha in the normal input structure).
        text : bool
            Whether or not to show the value as text in the plot.
        step : float, default=1
            The step size for the bins.   
        transparency : bool, default=False
            Change the transparency of the cell depending on the number of points in this range.
        interactive : bool, default=False
            Make interactive plots or not.
            
        See Also
        --------
        heatmap_brightest : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line.
        scatter_size : Generate scatter plots for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line with the size of the points depending on the number of data points in this range.
        
        """
        
        label = f'({param}_out - {param}_in)/{param}_in'
        
        # Check for outliers in the output A/N of all lines (keep within AoN max + 2) and filter accordingly
        # Where AoN <= 12, gives array of all 2D indices that have this. Remove any index that doesn't appear 8 times (peaks_no times)
        # Also remove the ones that go to -1
        ind = np.where(np.unique(np.argwhere(self.AoNs_out < 12)[:,1], return_counts=True)[1] == self.peaks_no)[0]
        
        close_0 = self.find_not_fit(peak=line, param=param, ind=ind)
        
        interest_AoN = self.AoNs_out[line][ind]
        sum_AoN = np.sum(self.AoNs_out[:,ind], axis=0)
        interest_val = self.find_relative_error(peak=line, param=param)[0][ind]
    
        interest_AoN = np.delete(interest_AoN, close_0)
        sum_AoN = np.delete(sum_AoN, close_0)
        interest_val = np.delete(interest_val, close_0)
        
        x_vals = np.arange(np.floor(min(sum_AoN)), np.ceil(max(sum_AoN)), step)
        y_vals = np.arange(np.floor(min(interest_AoN)), np.ceil(max(interest_AoN)), step)

        self.no_points = np.empty((len(x_vals) - 1, len(y_vals) - 1))

        stds = []
        medians = []

        for i in range(1, len(x_vals)):
            ind_x = (sum_AoN < x_vals[i]) * (sum_AoN > x_vals[i] - step)

            for j in range(1, len(y_vals)):
                ind_y = (interest_AoN < y_vals[j]) * (interest_AoN > y_vals[j] - step)
                
                self.no_points[i-1][j-1] = len(interest_AoN[ind_y*ind_x])
                stds.append(np.std(interest_val[ind_y*ind_x]))
                medians.append(np.median(interest_val[ind_y*ind_x]))
        
        if transparency == True:
            self.no_points = np.log10(self.no_points )/ np.log10(self.no_points.max())
            self.no_points = np.nan_to_num(self.no_points, neginf=0)
        else:
            self.no_points = np.ones_like(self.no_points)


        stds = np.reshape(stds, (len(x_vals)-1, len(y_vals)-1))
        medians = np.reshape(medians, (len(x_vals)-1, len(y_vals)-1))
        
        fig, ax = plt.subplots()

        # Plot the standard deviations
        if value == 'std':
            ax = plt.gca()  
            ax.set_facecolor('white')
            pc = plt.pcolormesh(x_vals, y_vals, stds.T, cmap='inferno', alpha=self.no_points.T)
            plt.colorbar(pc)
            plt.title(f'Standard deviation of {label}')
            plt.xlabel('Sum of A/N of all lines')
            plt.ylabel(f'A/N of line {line}')
            # Text
            if text == True:
                for i in range(len(x_vals)-1):
                    for j in range(len(y_vals)-1):
                        if np.isnan(stds.T[j][i]):
                            pass
                        else:
                            plt.text(x_vals[i]+step/2, y_vals[j]+step/2, np.round(stds.T[j][i], 2), ha='center', va='center', color='w', fontsize='x-small')
                            
        # Plot the medians
        elif  value == 'median':
            ax = plt.gca()  
            ax.set_facecolor('black')
            try:
                norm = TwoSlopeNorm(vmin=max(medians[np.isfinite(medians)].min(), np.mean(medians[np.isfinite(medians)]) - 3*np.std(medians[np.isfinite(medians)])), vcenter=0, vmax=min(medians[np.isfinite(medians)].max(), np.mean(medians[np.isfinite(medians)]) + 3*np.std(medians[np.isfinite(medians)])))
            except:
                norm = TwoSlopeNorm(vcenter=0)
            pc = plt.pcolormesh(x_vals, y_vals, medians.T, norm=norm, cmap='seismic', alpha=self.no_points.T)
            plt.colorbar(pc)
            plt.title(f'Median of {label}')
            plt.xlabel('Sum of A/N of all lines')
            plt.ylabel(f'A/N of line {line}')
            # Text
            if text == True:
                for i in range(len(x_vals)-1):
                    for j in range(len(y_vals)-1):
                        if np.isnan(medians.T[j][i]) or np.isinf(medians.T[j][i]):
                            pass
                        else:
                            plt.text(x_vals[i]+step/2, y_vals[j]+step/2, np.round(medians.T[j][i], 3), ha='center', va='center', color='k', fontsize='x-small')
        plt.show()
        
        if interactive == True:
            # Store the relevant data globally for interactive stuff
            global line_of_interest, param_of_interest, brightest_of_interest, step_of_interest, show_text, transparent, arr_of_interest, value_of_interest
            line_of_interest = line
            param_of_interest = param
            brightest_of_interest = brightest
            step_of_interest = step
            show_text = text
            transparent = transparency
            value_of_interest = value
            if value == 'std':
                arr_of_interest = stds
            elif value == 'median':
                arr_of_interest = medians
            
            global current_plot, heatmap, scat_size, heatmap_sum
            current_plot = 'heatmap sum'
            heatmap = False
            scat_size = False
            heatmap_sum = True
            
            # Click handling
            fig.canvas.mpl_connect('button_press_event', self.on_click)

    def heatmap_brightest(self, param, line, value, brightest=4, text=True, step=1, transparency=False, interactive=False):
        """
        Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line.

        Parameters
        ----------
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to check for.
        line : int
            The line of interest.
        value : {'std', 'median'}
            Plot the standard deviations or the medians.
        brightest : int, default=4
            The brightest line (default corresponds to H alpha in the normal input structure).
        text : bool
            Whether or not to show the value as text in the plot.
        step : float, default=1
            The step size for the bins.
        transparency : bool, default=False
            Change the transparency of the cell depending on the number of points in this range.
        interactive : bool, default=False
            Make interactive plots or not.   
            
        See Also
        --------
        heatmap_sum : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the sum of the A/N of all lines.
        scatter_size : Generate scatter plots for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line with the size of the points depending on the number of data points in this range.
        
        """
        
        label = f'({param}_out - {param}_in)/{param}_in'
        
        # Check for outliers in the output A/N of all lines (keep within AoN max + 2) and filter accordingly
        # Where AoN <= 12, gives array of all 2D indices that have this. Remove any index that doesn't appear 8 times (peaks_no times)
        # Also remove the ones that go to -1
        ind = np.where(np.unique(np.argwhere(self.AoNs_out < 12)[:,1], return_counts=True)[1] == self.peaks_no)[0]
        
        close_0 = self.find_not_fit(peak=line, param=param, ind=ind)
        
        interest_AoN = self.AoNs_out[line][ind]
        brightest_AoN = self.AoNs_out[brightest][ind]
        interest_val = self.find_relative_error(peak=line, param=param)[0][ind]
    
        interest_AoN = np.delete(interest_AoN, close_0)
        brightest_AoN = np.delete(brightest_AoN, close_0)
        interest_val = np.delete(interest_val, close_0)
        
        x_vals = np.arange(np.floor(min(brightest_AoN)), np.ceil(max(brightest_AoN)), step)
        y_vals = np.arange(np.floor(min(interest_AoN)), np.ceil(max(interest_AoN)), step)

        stds = []
        medians = []
        
        self.no_points = np.empty((len(x_vals) - 1, len(y_vals) - 1))

        for i in range(1, len(x_vals)):
            ind_x = (brightest_AoN < x_vals[i]) * (brightest_AoN > x_vals[i] - step)

            for j in range(1, len(y_vals)):
                ind_y = (interest_AoN < y_vals[j]) * (interest_AoN > y_vals[j] - step)
                
                self.no_points[i-1][j-1] = len(interest_AoN[ind_y*ind_x])
                stds.append(np.std(interest_val[ind_y*ind_x]))
                medians.append(np.median(interest_val[ind_y*ind_x]))
        
        if transparency == True:
            self.no_points = np.log10(self.no_points)/ np.log10(self.no_points.max())
            self.no_points = np.nan_to_num(self.no_points, neginf=0)
        else:
            self.no_points = np.ones_like(self.no_points)


        stds = np.reshape(stds, (len(x_vals)-1, len(y_vals)-1))
        medians = np.reshape(medians, (len(x_vals)-1, len(y_vals)-1))
        
        fig, ax = plt.subplots()

        # Plot the standard deviations
        if value == 'std':
            ax = plt.gca()  
            ax.set_facecolor('white')
            pc = plt.pcolormesh(x_vals, y_vals, stds.T, cmap='inferno', alpha=self.no_points.T, vmax=min(stds[np.isfinite(stds)].max(), np.mean(stds[stds > 0]) + 3*np.std(stds[stds > 0])))
            plt.colorbar(pc)
            plt.title(f'Standard deviation of {label}')
            plt.xlabel(f'A/N of line {brightest}')
            plt.ylabel(f'A/N of line {line}')
            # Text
            if text == True:
                for i in range(len(x_vals)-1):
                    for j in range(len(y_vals)-1):
                        if np.isnan(stds.T[j][i]):
                            pass
                        else:
                            plt.text(x_vals[i]+step/2, y_vals[j]+step/2, np.round(stds.T[j][i], 3), ha='center', va='center', color='w', fontsize='x-small')

        # Plot the medians
        elif value == 'median':
            try:
                norm = TwoSlopeNorm(vmin=max(medians[np.isfinite(medians)].min(), np.mean(medians[np.isfinite(medians)]) - 3*np.std(medians[np.isfinite(medians)])), vcenter=0, vmax=min(medians[np.isfinite(medians)].max(), np.mean(medians[np.isfinite(medians)]) + 3*np.std(medians[np.isfinite(medians)])))
            except:
                norm = TwoSlopeNorm(vcenter=0)
            pc = plt.pcolormesh(x_vals, y_vals, medians.T, norm=norm, cmap='seismic', alpha=self.no_points.T)
            plt.colorbar(pc)
            ax = plt.gca()  
            ax.set_facecolor('black')
            plt.title(f'Median of {label}')
            plt.xlabel(f'A/N of line {brightest}')
            plt.ylabel(f'A/N of line {line}')
            # Text
            if text == True:
                for i in range(len(x_vals)-1):
                    for j in range(len(y_vals)-1):
                        if np.isnan(medians.T[j][i]) or np.isinf(medians.T[j][i]):
                            pass
                        else:
                            plt.text(x_vals[i]+step/2, y_vals[j]+step/2, np.round(medians.T[j][i], 2), ha='center', va='center', color='k', fontsize='x-small')
        plt.show()
        
        if interactive == True:
            # Store the relevant data globally for interactive stuff
            global line_of_interest, param_of_interest, brightest_of_interest, step_of_interest, show_text, transparent, arr_of_interest, value_of_interest
            line_of_interest = line
            param_of_interest = param
            brightest_of_interest = brightest
            step_of_interest = step
            show_text = text
            transparent = transparency
            value_of_interest = value
            if value == 'std':
                arr_of_interest = stds
            elif value == 'median':
                arr_of_interest = medians
            
            global current_plot, heatmap, scat_size, heatmap_sum
            current_plot = 'heatmap'
            heatmap = True
            scat_size = False
            heatmap_sum = False
            
            # Click handling
            fig.canvas.mpl_connect('button_press_event', self.on_click)
            
    def plot_slice(self, lines, param, bright_l, bright_u, interest_l, interest_u, xlim=[-0.2, 11], ylim=[-5, 5], interactive=False):
        """
        Plot the difference between the input and output values of different components for a certain slice only.
        
        Parameters
        ----------
        lines : array
            Which lines of the spectrum to plot for.
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to plot for.
        bright_l : float
            The lower bound to the brightest line A/N.
        bright_u : float
            The upper bound to the brightest line A/N.
        interest_l : float
            The lower bound to the line of interest A/N.
        interest_u : float
            The upper bound to the line of interest A/N.
        xlim : array or None, default=[-0.2,11]
            The xlimits of the plot.
        ylim : array or None, default=[-5,5]
            The ylimits of the plot.
        interactive : bool, default=False
            Make interactive plots or not.
        """
 
        if param == 'sig':
            array = (self.sig_out - self.sig_in) / self.sig_in
        elif param == 'vel':
            array = (self.vels_out - self.vels_in) / self.vels_in
        elif param == 'A':
            array = (self.As_out - self.As_in) / self.As_in
        elif param == 'flux':
            array = (self.f_out - self.f_in) / self.f_in
        
        # Store line and array globally for click selecting 
        
        
        # Get the points of interest
        ind_int = (self.AoNs_out[lines[0]] < interest_u) * (self.AoNs_out[lines[0]] > interest_l)
        ind_bright = (self.AoNs_out[lines[1]] < bright_u) * (self.AoNs_out[lines[1]] > bright_l)

        fig, ax = plt.subplots(2, 1)
        
        label = f'({param}_out - {param}_in)/{param}_in'
        fig.suptitle(f'{label} against A/N')
        fig.supylabel(label)
        fig.supxlabel('A/N')
        
        ax[0].scatter(self.AoNs_out[lines[0]], array[lines[0]], s=2, label='All data')
        ax[0].scatter(self.AoNs_out[lines[0]][ind_int*ind_bright], array[lines[0]][ind_int*ind_bright], s=2, label='Data points\nof interest')
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(ylim)
        ax[0].legend()
        
        ax[1].scatter(self.AoNs_out[lines[1]], array[lines[1]], s=2)
        ax[1].scatter(self.AoNs_out[lines[1]][ind_int*ind_bright], array[lines[1]][ind_int*ind_bright], s=2)
        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylim)
        
        for i in range(len(lines)):
            if lines[i] == 0:
                ax[i].set_ylabel(r'H$\beta$')
            elif lines[i] == 1:
                ax[i].set_ylabel(r'[O III] 5006.77 $\AA$')
            elif lines[i] == 2:
                ax[i].set_ylabel(r'[O III] 4958.83 $\AA$')
            elif lines[i] == 3:
                ax[i].set_ylabel(r'[N II] 6547.96 $\AA$')
            elif lines[i] == 4:
                ax[i].set_ylabel(r'H$\alpha$')
            elif lines[i] == 5:
                ax[i].set_ylabel(r'[N II] 6583.34 $\AA$')
            elif lines[i] == 6:
                ax[i].set_ylabel(r'[S II] 6716.31 $\AA$')
            elif lines[i] == 7:
                ax[i].set_ylabel(r'[S II] 6730.68 $\AA$')
            else:
                ax[i].set_ylabel(f'Line {lines[i]}')

        plt.show()
        
        if interactive == True:
            global line_of_interest, param_of_interest, brightest_of_interest, arr_of_interest
            line_of_interest = lines[0]
            param_of_interest = param
            brightest_of_interest = lines[1]
            arr_of_interest = array
            
            
            global current_plot
            current_plot = 'slice'
            
            global axis
            axis = ax
            
            fig.canvas.mpl_connect('button_press_event', self.on_click)
              
    def scatter_size(self, param, line, value, brightest=4, step=1, interactive=False):
        """
        Generate scatter plots for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line with the size of the points depending on the number of data points in this range.

        Parameters
        ----------
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to check for.
        line : int
            The line of interest, y-axis.
        value : {'std', 'median'}
            Plot the standard deviations or the medians.
        brightest : int, default=4
            The brightest line (default corresponds to H alpha in the normal input structure), x-axis.
        step : float, default=1
            The step size for the bins.
        interactive : bool, default=False
            Make interactive plots or not.
            
        See Also
        --------
        heatmap_sum : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the sum of the A/N of all lines.
        heatmap_brightest : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line.

        """
        
        label = f'({param}_out - {param}_in)/{param}_in'
        
        # Check for outliers in the output A/N of all lines (keep within AoN max + 2) and filter accordingly
        # Where AoN <= 12, gives array of all 2D indices that have this. Remove any index that doesn't appear 8 times (peaks_no times)
        # Also remove the ones that go to -1
        ind = np.where(np.unique(np.argwhere(self.AoNs_out < 12)[:,1], return_counts=True)[1] == self.peaks_no)[0]
        
        close_0 = self.find_not_fit(peak=line, param=param, ind=ind)
        
        interest_AoN = self.AoNs_out[line][ind]
        brightest_AoN = self.AoNs_out[brightest][ind]
        interest_val = self.find_relative_error(peak=line, param=param)[0][ind]
    
        interest_AoN = np.delete(interest_AoN, close_0)
        brightest_AoN = np.delete(brightest_AoN, close_0)
        interest_val = np.delete(interest_val, close_0)
        
        x_vals = np.arange(np.floor(min(brightest_AoN)), np.ceil(max(brightest_AoN)), step)
        y_vals = np.arange(np.floor(min(interest_AoN)), np.ceil(max(interest_AoN)), step)
        
        
        
        mesh = np.meshgrid(x_vals[:-1] + step/2, y_vals[:-1] + step/2)

        stds = []
        medians = []
        
        self.no_points = np.empty((len(x_vals) - 1, len(y_vals) - 1))

        for i in range(1, len(x_vals)):
            ind_x = (brightest_AoN < x_vals[i]) * (brightest_AoN > x_vals[i] - step)

            for j in range(1, len(y_vals)):
                ind_y = (interest_AoN < y_vals[j]) * (interest_AoN > y_vals[j] - step)
                
                self.no_points[i-1][j-1] = len(interest_AoN[ind_y*ind_x])
                stds.append(np.std(interest_val[ind_y*ind_x]))
                medians.append(np.median(interest_val[ind_y*ind_x]))

        stds = np.reshape(stds, (len(x_vals)-1, len(y_vals)-1))
        medians = np.reshape(medians, (len(x_vals)-1, len(y_vals)-1))
        
        fig, ax = plt.subplots()
        
        # Plot the standard deviations
        if value == 'std':
            ax = plt.gca()  
            ax.set_facecolor('white')
            plt.title(f'Standard deviation of {label}')
            plt.plot(x_vals, x_vals * self.line_ratios[line]/self.line_ratios[brightest], color='k', linestyle=':')
            plt.scatter(mesh[0], mesh[1], s=(self.no_points.T), c=stds.T, cmap='inferno', vmax=min(stds[np.isfinite(stds)].max(), np.mean(stds[stds > 0]) + 3*np.std(stds[stds > 0])))

        # Plot the medians
        elif value == 'median':
            # v_ext = np.max([np.abs(medians[np.isfinite(medians)].max()), np.abs(medians[np.isfinite(medians)].min())])
            # try:
            #     norm = TwoSlopeNorm(vmin=-v_ext, vcenter=0, vmax=v_ext)
            # except:
            #     norm = TwoSlopeNorm(vcenter=0)
            
            # norm = MidpointNormalise(medians.T, midpoint=0)
            ax = plt.gca()  
            ax.set_facecolor("black")
            plt.title(f'Median of {label}')

            norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5)
            
            
            plt.plot(x_vals, x_vals * self.line_ratios[line]/self.line_ratios[brightest], color='w', linestyle=':')
            plt.scatter(mesh[0], mesh[1], s=(self.no_points.T), c=medians.T, cmap='seismic')#, norm=norm)
        
        if brightest == 4:
            plt.xlabel(r'A/N of H$\alpha$ line')
            plt.ylabel(r'A/N of H$\beta$ line')
        elif brightest == 1:
            plt.xlabel(r'A/N of [O III] 5006.77 $\AA$ line')
            plt.ylabel(r'A/N of [O III] 4958.83 $\AA$ line')
        elif brightest == 5:
            plt.xlabel(r'A/N of [N II] 6583.34 $\AA$ line')
            plt.ylabel(r'A/N of [N II] 6547.96 $\AA$ line')
        elif brightest == 7:
            plt.xlabel(r'A/N of [S II] 6730.68 $\AA$ line')
            plt.ylabel(r'A/N of [S II] 6716.31 $\AA$ line')
        else:
            plt.xlabel(f'A/N of line {brightest}')
            plt.ylabel(f'A/N of line {line}')
        plt.colorbar()
        plt.vlines(x_vals, min(y_vals), max(y_vals), color='lightgrey', linestyles='dotted')
        plt.hlines(y_vals, min(x_vals), max(x_vals), color='lightgrey', linestyles='dotted')
        plt.xlim([min(x_vals)-0.5, max(x_vals)+0.5])
        plt.ylim([min(y_vals)-0.5, max(y_vals)+0.5])
        # plt.savefig(f'scatter_{param}_{value}_{brightest}_{line}.png', dpi=600)
        plt.show()
        
        if interactive == True:
            # Store the relevant data globally for interactive stuff
            global line_of_interest, param_of_interest, brightest_of_interest, step_of_interest, arr_of_interest, value_of_interest
            line_of_interest = line
            param_of_interest = param
            brightest_of_interest = brightest
            step_of_interest = step
            value_of_interest = value
            if value == 'std':
                arr_of_interest = stds
            elif value == 'median':
                arr_of_interest = medians
            
            global current_plot, heatmap, scat_size, heatmap_sum
            current_plot = 'scatter size'
            heatmap = False
            scat_size = True
            heatmap_sum = False
            
            fig.canvas.mpl_connect('button_press_event', self.on_click)
    
class MidpointNormalise(colours.Normalize):
    def __init__(self, data, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        self.data = data
        
        colours.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.min([np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] ), self.midpoint + 3*np.nanstd(self.data)])


        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        print(v_ext)
        return np.ma.masked_array(np.interp(value, x, y))