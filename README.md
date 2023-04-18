# PoLite
This library is designed to facilitate the analysis of far-infrared and 
submillimeter polarimetric data sets. 

The two instruments currently supported are HAWC+ on the Stratospheric 
Observatory for Infrared Astronomy (SOFIA) and POL-2 at the James Clerk Maxwell
Telescope (JCMT). This script may someday include ALMA, APEX, NIKA-2, Planck.

Main package dependencies:
    
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
    
Full Package dependencies: 

Package           Version
----------------- ---------
aplpy             2.1.0
asteval           0.9.29
astropy           5.2.2
astropy-healpix   0.7
asttokens         2.2.1
backcall          0.2.0
colorama          0.4.6
comm              0.1.3
contourpy         1.0.7
cycler            0.11.0
debugpy           1.6.7
decorator         5.1.1
executing         1.2.0
fonttools         4.39.3
future            0.18.3
imageio           2.27.0
ipykernel         6.22.0
ipython           8.12.0
jedi              0.18.2
jupyter_client    8.2.0
jupyter_core      5.3.0
kiwisolver        1.4.4
lazy_loader       0.2
lmfit             1.2.0
matplotlib        3.7.1
matplotlib-inline 0.1.6
nest-asyncio      1.5.6
networkx          3.1
numpy             1.24.2
packaging         23.1
parso             0.8.3
pickleshare       0.7.5
Pillow            9.5.0
pip               23.1
platformdirs      3.2.0
prompt-toolkit    3.0.38
psutil            5.9.5
pure-eval         0.2.2
PyAVM             0.9.5
pyerfa            2.0.0.3
Pygments          2.15.0
pyparsing         3.0.9
pyregion          2.2.0
python-dateutil   2.8.2
PyWavelets        1.4.1
pywin32           306
PyYAML            6.0
pyzmq             25.0.2
reproject         0.10.0
scikit-image      0.20.0
scipy             1.10.1
setuptools        63.2.0
shapely           2.0.1
six               1.16.0
stack-data        0.6.2
tifffile          2023.4.12
tornado           6.3
traitlets         5.9.0
uncertainties     3.1.7
wcwidth           0.2.6