#!/usr/bin/env python
"""
setup.py - workaround for wxPython 2.4.x

Usage:
    % python setup.py py2app
"""

import macholib_patch
from distutils.core import setup
import py2app
from glob import glob


APP = ['mmind_app.py'] #main file of your app
DATA_FILES = [
	('img', glob('img/*.png')),
	('icons', glob('icons/*.png'))
]

OPTIONS = {'argv_emulation': True, 
           'site_packages': True, 
           'arch': 'i386', 
           # 'iconfile': 'lan.icns', #if you want to add some ico
           'plist': {
                'CFBundleName': 'Mastermind',
                'CFBundleShortVersionString':'1.0.0', # must be in X.X.X format
                'CFBundleVersion': '1.0.0',
                'CFBundleDevelopmentRegion': 'English', #optional - English is default
                }   
          }
setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
