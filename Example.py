# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:21:28 2020

This code serves as an example on how to use the PoLite package.

The 'PoLite.py' and '30dor_hawc_d.fits' files must be in the same folder. 

@author: scoude
"""

# Importing the PoLite package
import PoLite as pol

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