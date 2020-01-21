# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:52:05 2019

@authors: scoude, msgordon

This library is designed to facilitate the analysis of far-infrared and 
submillimetre polarimetric data sets. 

The two instruments currently supported are HAWC+ on the Stratospheric 
Observatory for Infrared Astronomy (SOFIA) and POL-2 at the James Clerk Maxwell
Telescope (JCMT). Someday will include ALMA, APEX, NIKA-2, Planck. 
"""

# REMINDER: Python arrays are inverted [y,x] relative to IDL [x,y]

# =============================================================================
# Package dependencies
# =============================================================================

import numpy as np
from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt
##from astropy.wcs import WCS

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
    
    # Plotting procedure for polarization maps
    def polmap(self):
        # Initialization
        # 
        
        print()
        print('==================================')
        print('Plotting a pretty polarization map')
        print('==================================')
        print()
        
        #wcs = WCS(test.header)
        #im = plt.subplot(projection=wcs)
        #im = plt.imshow(test.I)
        #im.axes.invert_yaxis()
        
        return
    
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
    hawc_obs.units = 'mJy per square arcsecond' # Setting working units
    conv = 1000.0*hawc_obs.pixel**-2.0 # Conversion factor from Jy/pixel
    
    # Loading data for every attribute of hawc_obs
    hawc_obs.I = conv*hawc_data[0].data # Stokes I
    hawc_obs.dI = conv*hawc_data[1].data # Uncertainty dI for Stokes I
    hawc_obs.Q = conv*hawc_data[2].data # Stokes Q
    hawc_obs.dQ = conv*hawc_data[3].data # Uncertainty dQ for Stokes Q
    hawc_obs.U = conv*hawc_data[4].data # Stokes U
    hawc_obs.dU = conv*hawc_data[5].data # Uncertainty dU for Stokes U
    hawc_obs.P = hawc_data[8].data # Polarization fraction P (debiased)
    hawc_obs.dp = hawc_data[9].data # Uncertainty dP for polarization fraction P
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
    print()
    return hawc_obs