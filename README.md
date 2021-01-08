# PoLite
This library is designed to facilitate the analysis of far-infrared and 
submillimeter polarimetric data sets. 

The two instruments currently supported are HAWC+ on the Stratospheric 
Observatory for Infrared Astronomy (SOFIA) and POL-2 at the James Clerk Maxwell
Telescope (JCMT). This script may someday include ALMA, APEX, NIKA-2, Planck.

Package dependencies:
    
    math
    matplotlib
    numpy
    scipy
    astropy
    aplpy
    lmfit

If using conda, you may have to install reproject 0.7.1 manually using the
following command:
    
    conda install -c astropy reproject

Acknowledgements:
    
    This project made use of APLpy, an open-source plotting package for Python
    (Robitaille and Bressert, 2012).
    
    This project made use of Astropy,\footnote{http://www.astropy.org} a 
    community-developed core Python package for Astronomy \citep{astropy:2013, 
    astropy:2018}.
    
    This project was conducted in part at the SOFIA Science Center,
    which is operated by the Universities Space Research Association 
    under contract NNA17BF53C with the National Aeronautics and 
    Space Administration.
    
