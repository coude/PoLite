# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:52:05 2019

@authors: scoude, msgordon

This library is designed to facilitate the analysis of far-infrared and 
submillimetre polarimetric data sets. 

The two instruments currently supported are HAWC+ on the Stratospheric 
Observatory for Infrared Astronomy (SOFIA) and POL-2 at the James Clerk Maxwell
Telescope (JCMT). This script will someday include ALMA, APEX, NIKA-2, Planck.

Acknowledgements:
    This research made use of APLpy, an open-source plotting package for Python
    (Robitaille and Bressert, 2012).
    This research made use of Astropy,\footnote{http://www.astropy.org} a 
    community-developed core Python package for Astronomy \citep{astropy:2013, 
    astropy:2018}.
    
"""

# REMINDER: Python arrays are inverted [y,x] relative to IDL [x,y]

# =============================================================================
# Package dependencies
# =============================================================================

import numpy as np
from astropy.io import fits
from astropy import wcs
from aplpy import FITSFigure

# =============================================================================
# Object containing the polarization data and important ancillary information
# =============================================================================
class obs:
    # Initial attributes of the GREAT array
    def __init__(self):
        # Polarization data
        self.I = np.empty([2,2]) # Stokes I
        self.dI = np.empty([2,2]) # Uncertainty dI for Stokes I
        self.Q = np.empty([2,2]) # Stokes Q
        self.dQ = np.empty([2,2]) # Uncertainty dQ for Stokes Q
        self.U = np.empty([2,2]) # Stokes U
        self.dU = np.empty([2,2]) # Uncertainty dU for Stokes U
        self.PI = np.empty([2,2]) # Polarized intensity PI (debiased)
        self.dPI = np.empty([2,2]) # Uncertainty dPI for polarized intensity PI
        self.P = np.empty([2,2]) # Polarization fraction P (debiased)
        self.dP = np.empty([2,2]) # Uncertainty dP for polarization fraction P
        self.O = np.empty([2,2]) # Polarization angle O
        self.B = np.empty([2,2]) # Rotated polarization angle B (+90 degrees)
        self.dO = np.empty([2,2]) # Uncertainty dO for polarization angle O
        
        # Observation properties
        self.instrument = 'None' # Instrument used for observations
        self.wavelength = 'None' # Wavelength observed
        self.beam = 'Unknown' # Resolution of the observations (Beam FWHM in arcseconds)
        self.pixel = 'Unknown' # Pixel scale of the data
        self.units = 'Unknown' # Units of the observations
        self.object = 'None' # Astronomical target name
        self.astrometry = 'None' # Astrometry information for the data
        self.header = 'None' # Copy of the main fits header
        self.size = np.empty([2]) # Size of the array in Y and X
        
# =============================================================================
# Plotting procedure for polarization maps
# =============================================================================
    def polmap(self, idi=50.0, pdp=3.0, color='rainbow', scalevec=0.4, 
               clevels=100, imin=0.0, imax=None, size_x = 9.0, size_y = 9.0,
               dpi = 100.0):
        # Initialization
        # idi : Signal-to-noise ratio for Stokes I total intensity. Default is
        #       idi = 50.0, which leads to an upper limit of at least dP = 5.0%
        #       for the error on the polarization fraction P.
        # pdp : Signal-to-noise ratio for the de-biased polarization fraction P.
        #       Default is pdp = 3.0, which is the commonly accepted detection
        #       threshold in the litterature. 
        # color : Color scheme used for the plots. Default is rainbow. 
        # scalevec : Length of the vectors plotted on the map.
        # clevels : Number of contours plotted on the contour plot.
        # imin : Minimum value plotted in the Stokes I intensity map.
        #        Default is 0.0.
        # imax : Maximum value plotted in the Stokes I intensity map.
        #        If no value is provided, maximum is automatically detected.
        # size_x : Width of the figure in inches.
        # size_y : Height of the figure in inches.
        # dpi : Number of pixels per inch for the figure. 
        
        print()
        print('======================================')
        print('Plotting a delightful polarization map')
        print('======================================')
        print()
        
        # Creating a new fits object for APLpy's FITSFigure method to recognize
        # for the Stokes I total intensity
        plot_hdu = fits.PrimaryHDU(data=self.I, header=self.header)
        # Loading the data in an APLpy figure 
        polmap = FITSFigure(plot_hdu, dpi=dpi, figsize=(size_x, size_y))
        
        print('Plotting the Stokes I total intensity map')
        print()
        
        # Showing the pixelated image and setting the aspect ratio
        polmap.show_colorscale(cmap=color, vmin=imin, vmax=imax)
        # Plotting a filled contour plot of the data
        polmap.show_contour(cmap=color, levels=clevels, filled=True, 
                            extend='both')
        
        # Adding a colorbar to the plot
        polmap.add_colorbar(pad=0.125)
        # Adding the units to the colorbar
        polmap.colorbar.set_axis_label_text(self.units)
        # Moving the colorbar to be on top of the figure
        polmap.colorbar.set_location('top')
        
        # Creating a new fits object for APLpy's FITSFigure method to recognize
        # for the polarization fraction P
        pmap = fits.PrimaryHDU(data=self.P, header=self.header)
        # for the polarization angle O
        omap = fits.PrimaryHDU(data=self.O, header=self.header)
        
        # Muting the numpy alerts triggered by nan values
        np.warnings.filterwarnings('ignore')
        
        print()
        print('Applying happy selection criteria on polarization vectors')
        print()
        
        # Creating a mask to hide polarization vectors with low signal-to-noise ratios
        imask_01 = np.where(self.I/self.dI < idi) # Total intensity SNR threshold
        imask_02 = np.where(self.I < 0.0) # Total intensity positive threshold
        pmask = np.where(self.P/self.dP < pdp) # Polarization SNR threshold
        # Masking all the indices for which the selection criteria failed
        pmap.data[imask_01] = np.nan
        pmap.data[imask_02] = np.nan
        pmap.data[pmask] = np.nan
        
        print('Plotting the polarization vectors')
        print()
        
        # Plotting the polarization vectors
        polmap.show_vectors(pmap, omap, scale=scalevec)
        
        # Adding the beam size
        polmap.add_beam(facecolor='white', edgecolor='black',
                      linewidth=2, pad=1, corner='bottom left')
        
        # Adding the vector scale bar
        vectscale = scalevec * self.pixel/3600
        polmap.add_scalebar(5 * vectscale, "p = 5%",corner='top right',frame=True)
        
        # Removing the pixelated structure under the figure
        polmap.hide_colorscale() 
        
        print('Returning the APLpy figure as output, please feel free to' +
              'improve it (see online APLpy.FITSFigure documentation)')
        print()
        print('Don\'t forget to save the results using the' + 
              ' .save(\'name.png\') function')
        print()
        print('Don\'t give up!')
        
        return polmap

# =============================================================================
# Plotting procedure for magnetic field maps
# =============================================================================
    def Bmap(self, idi=50.0, pdp=3.0, color='rainbow', scalevec=1.0,
               clevels=100, imin=0.0, imax=None, size_x = 9.0, size_y = 9.0, 
               dpi = 100.0):
        # Initialization
        # idi : Signal-to-noise ratio for Stokes I total intensity. Default is
        #       idi = 50.0, which leads to an upper limit of at least dP = 5.0%
        #       for the error on the polarization fraction P.
        # pdp : Signal-to-noise ratio for the de-biased polarization fraction P.
        #       Default is pdp = 3.0, which is the commonly accepted detection
        #       threshold in the litterature. 
        # color : Color scheme used for the plots. Default is rainbow. 
        # scalevec : Length of the vectors plotted on the map.
        # clevels : Number of contours plotted on the contour plot.
        # imin : Minimum value plotted in the Stokes I intensity map.
        #        Default is 0.0.
        # imax : Maximum value plotted in the Stokes I intensity map.
        #        If no value is provided, maximum is automatically detected.
        
        print()
        print('=====================================')
        print('Plotting a lovable magnetic field map')
        print('=====================================')
        print()
        
        # Creating a new fits object for APLpy's FITSFigure method to recognize
        # for the Stokes I total intensity
        plot_hdu = fits.PrimaryHDU(data=self.I, header=self.header)
        # Loading the data in an APLpy figure 
        Bmap = FITSFigure(plot_hdu, dpi=dpi, figsize=(size_x, size_y))
        
        print('Plotting the Stokes I total intensity map')
        print()
        
        # Showing the pixelated image and setting the aspect ratio
        Bmap.show_colorscale(cmap=color, vmin=imin, vmax=imax)
        # Plotting a filled contour plot of the data
        Bmap.show_contour(cmap=color, levels=clevels, filled=True, 
                          extend='both', vmin=imin, vmax=imax)
        # Inverting ticks
        #Bmap.ticks.set_tick_direction('in') # BUGGED in APLpy
        
        # Adding a colorbar to the plot
        Bmap.add_colorbar(pad=0.125)
        # Adding the units to the colorbar
        Bmap.colorbar.set_axis_label_text(self.units)
        # Moving the colorbar to be on top of the figure
        Bmap.colorbar.set_location('top')
        
        # Creating a new fits object for APLpy's FITSFigure method to recognize
        # for the polarization fraction P
        pmap = fits.PrimaryHDU(data=self.P, header=self.header)
        # for the polarization angle O
        omap = fits.PrimaryHDU(data=self.B, header=self.header)
        
        # Muting the numpy alerts triggered by nan values
        np.warnings.filterwarnings('ignore')
        
        print()
        print('Applying happy selection criteria on polarization vectors')
        print()
        
        # Creating a mask to hide polarization vectors with low signal-to-noise ratios
        imask_01 = np.where(self.I/self.dI < idi) # Total intensity SNR threshold
        imask_02 = np.where(self.I < 0) # Total intensity positive threshold
        pmask = np.where(self.P/self.dP < pdp) # Polarization SNR threshold
        # Forcing the polarization vectors to share the same amplitude
        pmap.data[np.where(self.P > 0.0)] = 1.0
        # Masking all the indices for which the selection criteria failed
        pmap.data[imask_01] = np.nan
        pmap.data[imask_02] = np.nan
        pmap.data[pmask] = np.nan
        
        print('Plotting the magnetic field segments')
        print()
        
        # Plotting the polarization vectors
        Bmap.show_vectors(pmap, omap, scale=scalevec)
        
        # Adding the beam size
        Bmap.add_beam(facecolor='white', edgecolor='black',
                      linewidth=2, pad=1, corner='bottom left')
        
        # Removing the pixelated structure under the figure
        Bmap.hide_colorscale() # Warning: You need to reopen it to create a new
                                        # colorbar, APLpy deals poorly with
                                        # contour maps by themselves
        
        print('Returning the APLpy figure as output, please feel free to' +
              'improve it (see online APLpy.FITSFigure documentation)')
        print()
        print('Don\'t forget to save the results using the' + 
              ' .save(\'name.png\') function')
        print()
        print('Have fun!')
        
        return Bmap
    
# =============================================================================
# Function to create an obs object from a HAWC+ data cube
# =============================================================================
def load_hawc(fits_name):
    # Initialization
    # fits_name : String for the name of the fits file to be loaded
         
    # Loading the HAWC+ fits file
    print()
    print('=========================')
    print('Kindly loading HAWC+ data')
    print('=========================')
    print()
    print('Opening FITS file:')
    print(fits_name)
    print()
    hawc_data = fits.open(fits_name)
    # Creating the obs object to be provided by the function
    hawc_obs = obs()
    
    # Loading the ancillary information for hawc_obs
    hawc_obs.header = hawc_data[0].header # Copy of the primary fits header
    hawc_obs.instrument = hawc_data[0].header['INSTRUME'] # Instrument used for observations
    hawc_obs.wavelength = hawc_data[0].header['WAVECENT'] # Wavelength observed
    hawc_obs.object = hawc_data[0].header['OBJECT'] # Astronomical target name
    hawc_obs.astrometry = wcs.WCS(hawc_data[0].header) # Astrometry information for the data
    hawc_obs.beam = (hawc_data[0].header['BMAJ'])*3600.0 # Beam size in arcseconds
    
    print('Astronomical object: '+hawc_obs.object)
    print('Wavelength observed: '+str(hawc_obs.wavelength)+' µm')
    print()
    
    # Unit conversion from Jy/pixel to mJy/arcsec^2
    hawc_obs.pixel = (hawc_data[0].header['PIXSCAL']) # Pixel scale of the data in arcseconds
    hawc_obs.units = 'mJy per arcsec$^2$' # Setting working units
    conv = 1000.0*hawc_obs.pixel**-2.0 # Conversion factor from Jy/pixel
    
    # Loading data for every attribute of hawc_obs
    hawc_obs.I = conv*hawc_data[0].data # Stokes I
    hawc_obs.dI = conv*hawc_data[1].data # Uncertainty dI for Stokes I
    hawc_obs.Q = conv*hawc_data[2].data # Stokes Q
    hawc_obs.dQ = conv*hawc_data[3].data # Uncertainty dQ for Stokes Q
    hawc_obs.U = conv*hawc_data[4].data # Stokes U
    hawc_obs.dU = conv*hawc_data[5].data # Uncertainty dU for Stokes U
    hawc_obs.P = hawc_data[8].data # Polarization fraction P (debiased)
    hawc_obs.dP = hawc_data[9].data # Uncertainty dP for polarization fraction P
    hawc_obs.O = hawc_data[10].data # Polarization angle O
    hawc_obs.B = hawc_data[11].data # Rotated polarization angle B (+90 degrees)
    hawc_obs.dO = hawc_data[12].data # Uncertainty dO for polarization angles
    hawc_obs.PI = conv*hawc_data[13].data # Polarized intensity PI (debiased)
    hawc_obs.dPI = conv*hawc_data[14].data # Uncertainty dPI for polarized intensity PI
    
    # Finding the Y and X size of the arrays 
    hawc_obs.size = hawc_obs.I.shape # hawc_obs.size should be in the form (Y,X)

    # Closing access to the fits file
    hawc_data.close()
    
    # Returning the obs object created with the HAWC+ data cube
    print('Have a nice day!')
    return hawc_obs