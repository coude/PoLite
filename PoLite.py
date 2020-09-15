# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:52:05 2019

@authors: scoude, msgordon

This library is designed to facilitate the analysis of far-infrared and 
submillimeter polarimetric data sets. 

The two instruments currently supported are HAWC+ on the Stratospheric 
Observatory for Infrared Astronomy (SOFIA) and POL-2 at the James Clerk Maxwell
Telescope (JCMT). This script may someday include ALMA, APEX, NIKA-2, Planck.

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
import math
#import warnings

# Warning filter
#warnings.simplefilter('ignore',category=wcs.FITSFixedWarning)
# The following mutes warnings about NaN values
np.warnings.filterwarnings('ignore')

# =============================================================================
# Object containing the polarization data and important ancillary information
# =============================================================================
class cat:
    # Default attributes of the object
    def __init__(self):
        # Polarization data
        self.I = np.empty([2]) # Stokes I
        self.dI = np.empty([2]) # Uncertainty dI for Stokes I
        self.Q = np.empty([2]) # Stokes Q
        self.dQ = np.empty([2]) # Uncertainty dQ for Stokes Q
        self.U = np.empty([2]) # Stokes U
        self.dU = np.empty([2]) # Uncertainty dU for Stokes U
        self.PI = np.empty([2]) # Polarized intensity PI (debiased)
        self.dPI = np.empty([2]) # Uncertainty dPI for polarized intensity PI
        self.P = np.empty([2]) # Polarization fraction P (debiased)
        self.dP = np.empty([2]) # Uncertainty dP for polarization fraction P
        self.O = np.empty([2]) # Polarization angle O
        self.B = np.empty([2]) # Rotated polarization angle B (+90 degrees)
        self.dO = np.empty([2]) # Uncertainty dO for polarization angle O
        self.RA = np.empty([2]) # Right Ascension coordinates in degrees
        self.DEC = np.empty([2]) # Declination coordinates in degrees

# =============================================================================
# Object containing the polarization data and important ancillary information
# =============================================================================
class obs:
    # Default attributes of the object
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
        self.RA = np.empty([2,2]) # Right Ascension coordinates in degrees
        self.DEC = np.empty([2,2]) # Declination coordinates in degrees
        
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
    def polmap(self, idi=50.0, pdp=3.0, pmax=30.0, color='rainbow', 
               scalevec=0.4, clevels=100, imin=0.0, imax=None, 
               size_x = 9.0, size_y = 9.0, dpi = 100.0):
        # Initialization
        # idi : Signal-to-noise ratio for Stokes I total intensity. Default is
        #       idi = 50.0, which leads to an upper limit of at least dP = 5.0%
        #       for the error on the polarization fraction P.
        # pdp : Signal-to-noise ratio for the de-biased polarization fraction P.
        #       Default is pdp = 3.0, which is the commonly accepted detection
        #       threshold in the litterature. 
        # pmax : Maximum polarization fraction allowed.
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
                            extend='both', vmin=imin, vmax=imax)
        
        # Adding a colorbar to the plot
        polmap.add_colorbar(pad=0.125)
        # Adding the units to the colorbar
        polmap.colorbar.set_axis_label_text(self.units)
        # Moving the colorbar to be on top of the figure
        polmap.colorbar.set_location('top')
        
        # Creating a new fits object for APLpy's FITSFigure method to recognize
        # for the polarization fraction P
        pref = fits.PrimaryHDU(data=self.P, header=self.header)
        pmap = pref.copy()
        # for the magnetic field angle B
        oref = fits.PrimaryHDU(data=self.O, header=self.header)
        omap = oref.copy()
        # A copy of the Primary HDU is created before any masking is done.
        # This fixes an issue where modifying pmap was carried to other
        # instances of this method. Future fix should make sure pref is
        # closed/deleted at the end of this method.
        
        # Muting the numpy alerts triggered by nan values
        np.warnings.filterwarnings('ignore')
        
        print()
        print('Applying happy selection criteria on polarization vectors')
        print()
        
        # Creating a mask to hide polarization vectors with low signal-to-noise ratios
        imask_01 = np.where(self.I/self.dI < idi) # Total intensity SNR threshold
        imask_02 = np.where(self.I < 0.0) # Total intensity positive threshold
        pmask = np.where(self.P/self.dP < pdp) # Polarization SNR threshold
        pmaxmask = np.where(self.P > pmax) # Polarization SNR threshold
        # Masking all the indices for which the selection criteria failed
        pmap.data[imask_01] = np.nan
        pmap.data[imask_02] = np.nan
        pmap.data[pmask] = np.nan
        pmap.data[pmaxmask] = np.nan
        
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
              ' improve it (see online APLpy.FITSFigure documentation)')
        print()
        print('Don\'t forget to save the results using the' + 
              ' .savefig(\'name.png\') function')
        print()
        print('Don\'t give up!')
        
        return polmap

# =============================================================================
# Plotting procedure for magnetic field maps
# =============================================================================
    def Bmap(self, idi=50.0, pdp=3.0, pmax=30.0, color='rainbow', scalevec=1.0,
               clevels=100, imin=0.0, imax=None, size_x = 9.0, size_y = 9.0, 
               dpi = 100.0):
        # Initialization
        # idi : Signal-to-noise ratio for Stokes I total intensity. Default is
        #       idi = 50.0, which leads to an upper limit of at least dP = 5.0%
        #       for the error on the polarization fraction P.
        # pdp : Signal-to-noise ratio for the de-biased polarization fraction P.
        #       Default is pdp = 3.0, which is the commonly accepted detection
        #       threshold in the litterature. 
        # pmax : Maximum polarization fraction allowed.
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
        Bmap = FITSFigure(plot_hdu, dpi=dpi, figsize=(size_x, size_y),
                          slices=[0,1])
        
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
        pref = fits.PrimaryHDU(data=self.P, header=self.header)
        pmap = pref.copy()
        # for the magnetic field angle B
        oref = fits.PrimaryHDU(data=self.B, header=self.header)
        omap = oref.copy()
        # A copy of the HDU is created before any masking is done.
        # This fixes an issue where modifying pmap was carried to other
        # instances of this method. Future fix should make sure pref is
        # closed/deleted at the end of this method.
                
        # Muting the numpy alerts triggered by nan values
        np.warnings.filterwarnings('ignore')
        
        print()
        print('Applying happy selection criteria on polarization vectors')
        print()
        
        # Creating a mask to hide polarization vectors with low signal-to-noise ratios
        imask_01 = np.where(self.I/self.dI < idi) # Total intensity SNR threshold
        imask_02 = np.where(self.I < 0) # Total intensity positive threshold
        pmask = np.where(self.P/self.dP < pdp) # Polarization SNR threshold
        pmaxmask = np.where(self.P > pmax) # Polarization SNR threshold
        # Forcing the polarization vectors to share the same amplitude
        pmap.data[np.where(self.P > 0.0)] = 1.0
        # Masking all the indices for which the selection criteria failed
        pmap.data[imask_01] = np.nan
        pmap.data[imask_02] = np.nan
        pmap.data[pmask] = np.nan
        pmap.data[pmaxmask] = np.nan
        
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
              ' improve it (see online APLpy.FITSFigure documentation)')
        print()
        print('Don\'t forget to save the results using the' + 
              ' .savefig(\'name.png\') function')
        print()
        print('Have fun!')
        
        return Bmap

# =============================================================================
# Creating a mask for a region of the map
# =============================================================================
    def MaskRegion(self, shape=None, center_RA=None, center_DEC=None, 
                   radius=None, width=None, height=None):
        # Method to create a simple mask (either circular or rectangular) for 
        # data analysis.
        
        # Initialization
        # shape : Shape of the region masked, either "Circle" or "Rectangle".
        # center_RA : Central position of the mask in right ascension (degrees)
        # center_DEC : Central position of the mask in declination (degrees)
        # radius: If shape is Circle, then this paramter provides its radius (degrees)
        # width: If shape is Rectangle, then this paramter provides its width in right ascencion (degrees)
        # height: If shape is Circle, then this paramter provides its heigh in declination (degrees)
        
        print()
        print('=============================')
        print('Respectfully masking a region')
        print('=============================')
        print()
        
        # Creating the empty array for the final mask
        mask = fits.PrimaryHDU(data=np.zeros(self.size), header=self.header)
        
        # Checking the shape for the region
        if shape == 'Circle':
            print('Creating a circular mask')
            print()
        elif shape == 'Rectangle':
            print('Creating a rectangular mask')
            print()
        else:
            print('Shape should be "Circle" or "Rectangle"')
            return mask
        
        # Checking the coordinates of the central posituon
        if center_RA == None or center_DEC == None:
            print('Please provide the center_RA and center_DEC parameters '+
                  'of the central position in degrees')
            return mask
        
        # Checking the parameters for the plotted shape
        if shape == 'Circle' and radius == None:
            print('Please provide the radius of the circle in degrees')
            return mask
        if shape == 'Rectangle' and (width == None or height == None):
            print('Please provide the length and height of the rectangle '+
                  'in degrees')
            return mask
        
        # If the shape is a Circle
        if shape == 'Circle':            
            # Identifying pixels to be included in the mask
            for i in range (0,self.size[0]): # Declination
                for j in range (0,self.size[1]): # Right Ascension
                    # Distance to central pixel
                    delta = (((self.RA[i,j] - 
                               center_RA)*math.cos(self.DEC[i,j]*math.pi/180.0
                                                   ))**2.0 + 
                              (self.DEC[i,j] - center_DEC)**2.0)**0.5
                    # Is the pixel in the mask?
                    if delta < radius:
                        mask.data[i,j] = 1.0
                    else:
                        mask.data[i,j] = np.nan
        
        # If the shape is a Rectangle
        if shape == 'Rectangle':            
            # Identifying pixels to be included in the mask
            for i in range (0,self.size[0]): # Declination
                for j in range (0,self.size[1]): # Right Ascension
                    # Distance to central pixel
                    delta_ra = abs((self.RA[i,j] - 
                               center_RA) * math.cos(
                                   self.DEC[i,j]*math.pi/180.0))
                    delta_dec = abs(self.DEC[i,j] - center_DEC)

                    # Is the pixel in the mask?
                    if delta_ra < 0.5*width and delta_dec < 0.5*height:
                        mask.data[i,j] = 1.0
                    else:
                        mask.data[i,j] = np.nan
        
        print('Providing requested mask, have fun!')
        return mask
    
# =============================================================================
# Creating a mask for a region of the map
# =============================================================================
    def CombineMasks(self, mask1, mask2):
        # Method to combine two masks into one for data analysis
        
        # Initialization
        # mask1, mask2 : Fits files for the masks to be combined
        
        print()
        print('========================')
        print('Combining friendly masks')
        print('========================')
        print()
        
        # Size of the maps
        size1 = np.shape(mask1.data)
        size2 = np.shape(mask2.data)
        size_ref = self.size
        
        # Creating the empty array for the final mask
        mask = fits.PrimaryHDU(data=np.zeros(self.size), header=self.header)
        
        # Check that both masks have the same size as the obs object
        if size1 != size_ref or size2 != size_ref:
            print('Masks must have the same shape as the data to be analyzed')
            return mask
        
        # Creating the combined mask
        mask.data[np.where(mask.data < 1)] = np.nan
        mask.data[np.where(mask1.data > 0)] = 1.0  
        mask.data[np.where(mask2.data > 0)] = 1.0  

        print('Providing combined mask, good luck!')
        return mask
    
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
    hawc_obs.wavelength = hawc_data[0].header['WAVECENT'] # Wavelength observed in microns
    hawc_obs.object = hawc_data[0].header['OBJECT'] # Astronomical target name
    hawc_obs.astrometry = wcs.WCS(hawc_data[0].header) # Astrometry information for the data
    hawc_obs.beam = (hawc_data[0].header['BMAJ'])*3600.0 # Beam size in arcseconds
    
    print('Astronomical object: '+hawc_obs.object)
    print('Wavelength observed: '+str(hawc_obs.wavelength)+' µm')
    print()
    
    # Unit conversion from Jy/pixel to mJy/arcsec^2
    hawc_obs.pixel = 3600.0*(hawc_data[0].header['CDELT2']) # Pixel scale of the data in arcseconds
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
    
    # Expanding the arrays of coordinates
    hawc_obs.RA = np.zeros(hawc_obs.size)
    hawc_obs.DEC = np.zeros(hawc_obs.size)
    # Identifying the world coordinates for each pixel - NOT EFFICIENT?    
    for i in range (0,hawc_obs.size[0]): # Declination
        for j in range (0,hawc_obs.size[1]): # Right Ascension
            hawc_obs.RA[i,j], hawc_obs.DEC[i,j] = (
                hawc_obs.astrometry.array_index_to_world_values(i,j))

    # Closing access to the fits file
    hawc_data.close()
    
    # Returning the obs object created with the HAWC+ data cube
    print('Have a nice day!')
    return hawc_obs

# =============================================================================
# Function to create an obs object from POL-2 Stokes I, Q, and U data
# =============================================================================
def load_pol2(target, fits_imap, fits_qmap, fits_umap):
    # Initialization
    # target : String for the target name
    # fits_imap : String for the name of the Stokes I map to be loaded
    # fits_qmap : String for the name of the Stokes Q map to be loaded
    # fits_umap : String for the name of the Stokes U map to be loaded
    
    # This function is valid as of 2020/05/17
           
    print()
    print('=============================')
    print('Cheerfully loading POL-2 data')
    print('=============================')
    print()
    print('Opening Stokes I file:')
    print(fits_imap)
    print()
    print('Opening Stokes Q file:')
    print(fits_qmap)
    print()
    print('Opening Stokes U file:')
    print(fits_umap)
    print()

    # Loading the POL-2 fits file
    pol2_imap = fits.open(fits_imap)
    pol2_qmap = fits.open(fits_qmap)
    pol2_umap = fits.open(fits_umap)
    # Creating the obs object to be provided by the function
    pol2_obs = obs()
    
    # Loading the ancillary information for pol2_obs
    pol2_obs.header = pol2_imap[0].header # Copy of the primary fits header
    pol2_obs.instrument = 'SCUBA-2 / POL-2' # Instrument used for observations
    pol2_obs.wavelength = 10.0**6.0*pol2_imap[0].header['WAVELEN'] # Wavelength observed in microns
    pol2_obs.object = target # Astronomical target name
    pol2_obs.pixel = 3600.0*(pol2_imap[0].header['CDELT2']) # Pixel scale in arcseconds
    pol2_obs.units = 'mJy per arcsec$^2$' # Units of the loaded data
    
    print('Astronomical object: ' + pol2_obs.object)
    print('Wavelength observed: ' + str(pol2_obs.wavelength) + ' µm')
    print()
    
    # Unit conversion from pW/beam to Jy/beam to mJy/arcsec^2
    # Beam size in arcseconds
    if pol2_obs.wavelength == 450.0:
        pol2_obs.beam = 9.8 # Effective beam size at 450 um in arcseconds
        conv = 1000.0*1.35 * 491.0 # As of 2020/01/23
    elif pol2_obs.wavelength == 850.0:
        pol2_obs.beam = 14.6 # Effective beam size at 850 um in arcseconds
        conv = 1000.0*1.35 * 537.0 # As of 2020/01/23
    else:
        print('Sorry, but no valid wavelength was provided...')
    
    # Loading data for every attribute of hawc_obs
    pol2_obs.I = np.squeeze(conv*pol2_imap[0].data[0]) # Stokes I
    pol2_obs.dI = np.squeeze(conv*pol2_imap[1].data[0]**0.5) # Uncertainty dI for Stokes I
    pol2_obs.Q = np.squeeze(conv*pol2_qmap[0].data[0]) # Stokes Q
    pol2_obs.dQ = np.squeeze(conv*pol2_qmap[1].data[0]**0.5) # Uncertainty dQ for Stokes Q
    pol2_obs.U = np.squeeze(conv*pol2_umap[0].data[0]) # Stokes U
    pol2_obs.dU = np.squeeze(conv*pol2_umap[1].data[0]**0.5) # Uncertainty dU for Stokes U
    
    # Adapting the header if a third dimension exists in the data
    # Ugly workaround, may need to be improved in the future
    pol2_obs.header['NAXIS'] = 2
    pol2_obs.header.remove('NAXIS3')
    pol2_obs.header.remove('LBOUND3')
    pol2_obs.header.remove('CRPIX3')
    pol2_obs.header.remove('CRVAL3')
    pol2_obs.header.remove('CDELT3')
    pol2_obs.header.remove('CTYPE3')
    pol2_obs.header.remove('CUNIT3')
    pol2_obs.header.remove('CRPIX3A')
    pol2_obs.header.remove('CRVAL3A')
    pol2_obs.header.remove('CDELT3A')
    pol2_obs.header.remove('CUNIT3A')
    pol2_obs.header.append(
            ('BMAJ',pol2_obs.beam/3600.0, 'Beam major axis'))
    pol2_obs.header.append(
            ('BMIN',pol2_obs.beam/3600.0, 'Beam minor axis'))
    pol2_obs.header.append(('BPA',0.0, 'Beam position angle'))
    pol2_obs.astrometry = wcs.WCS(pol2_obs.header) # Astrometry information for the data
    #pol2_obs.astrometry = pol2_obs.astrometry.dropaxis(2) # Potential fix, investigate
    
    
    # Finding the Y and X size of the arrays 
    pol2_obs.size = pol2_obs.I.shape # pol2_obs.size should be in the form (Y,X)
    
    # ======================================
    # Generating the polarization properties
    # ======================================
    
    # Biased polarized intensity
    pi_biased = (pol2_obs.Q**2.0 + pol2_obs.U**2.0)**0.5
    # PI uncertainties
    pol2_obs.dPI = ((pol2_obs.Q*pol2_obs.dQ)**2.0 + 
                    (pol2_obs.U*pol2_obs.dU)**2.0)**0.5/pi_biased
    # De-biased polarized intensity PI
    pol2_obs.PI = (pi_biased**2.0 - pol2_obs.dPI**2.0)**0.5
    # Removing unphysical negative values
    pimask = np.where(pol2_obs.PI < 0.0)
    imask = np.where(pol2_obs.I < 0.0)
    # Masking the values in PI and dPI
    pol2_obs.PI[pimask] = np.nan
    pol2_obs.dPI[pimask] = np.nan
    pol2_obs.PI[imask] = np.nan
    pol2_obs.dPI[imask] = np.nan
    
    # Polarization fraction P
    pol2_obs.P = 100.0*pol2_obs.PI/pol2_obs.I
    # Uncertainty of polarization fraction
    pol2_obs.dP = pol2_obs.P*((pol2_obs.dPI/pol2_obs.PI)**2.0 + 
                              (pol2_obs.dI/pol2_obs.I)**2.0)**0.5
    
    # Polarization angle O
    pol2_obs.O = (0.5*180.0/math.pi)*np.arctan2(pol2_obs.U, pol2_obs.Q)
    # Uncertainties dO
    pol2_obs.dO = (0.5*180.0/math.pi)*((pol2_obs.Q*pol2_obs.dU)**2.0 + 
                       (pol2_obs.U*pol2_obs.dQ)**2.0)**0.5/pi_biased
    # Polarization angle B
    pol2_obs.B = pol2_obs.O + 90.0
    pol2_obs.B[np.where(pol2_obs.B > 90.0)] = pol2_obs.B[np.where(pol2_obs.B 
               > 90.0)] - 180.0  
    
    # Expanding the arrays of coordinates
    pol2_obs.RA = np.zeros(pol2_obs.size)
    pol2_obs.DEC = np.zeros(pol2_obs.size)
    # Identifying the world coordinates for each pixel - NOT EFFICIENT?    
    for i in range (0,pol2_obs.size[0]): # Declination
        for j in range (0,pol2_obs.size[1]): # Right Ascension
            pol2_obs.RA[i,j], pol2_obs.DEC[i,j] = (
                pol2_obs.astrometry.array_index_to_world_values(i, j))
    
    # Closing access to the fits files
    pol2_imap.close()
    pol2_qmap.close()
    pol2_umap.close()
    
    # Returning the obs object created with the POL-2 data
    return pol2_obs












