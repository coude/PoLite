# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:21:28 2020

This code serves as an example on how to use the PoLite package.

The 'PoLite.py' and '30dor_hawc_d.fits' files must be in the same folder. 

@author: scoude
"""

# Importing the PoLite package
from PoLite import PoLite as pol

# Define the name for the file to be imported
hawc_file = '30dor_hawc_d.fits'

# Load the HAWC+ file into an "obs" object
polobs = pol.load_hawc(hawc_file)

# Creating polarization map
polmap = polobs.polmap()

# Create a magnetic field map
Bmap = polobs.Bmap()

# Saving the figures as PNG files
polmap.savefig('polmap.png',dpi=300)
Bmap.savefig('Bmap.png',dpi=300)

# Creating the vector catalog for the whole region
BigCat = polobs.MakeCat()

# Plotting the histogram and the Gaussian fit for the catalog
BigHisto, BigFit = BigCat.Histogram()
BigHisto.savefig('BigCat.png',dpi=300)

# Creating a mask near the middle of the map
ra_center = 84.69506551 # in degrees
dec_center = -69.08220438 # in degrees
ra_size = 0.01311784234  # in degrees
dec_size = 0.01311784234 # in degrees
PolMask = polobs.MaskRegion(shape='Rectangle', center_RA=ra_center, 
                            center_DEC=dec_center,width=ra_size,
                            height=dec_size)

# The MakeCat() function can use the mask above to create a smaller catalog
SmallCat = polobs.MakeCat(mask=PolMask)

# Plotting the histogram and the Gaussian fit for the smaller catalog
SmallHisto, SmallFit = SmallCat.Histogram()
SmallHisto.savefig('SmallCat.png',dpi=300)