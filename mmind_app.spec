# -*- mode: python -*-

block_cipher = None
path = '/Users/Matthias/Studium/WS 15/mastermind/code'

a = Analysis(['mmind_app.py', 'mastermind.py', 'sharma_mittal.py'],
             pathex=[path],
             binaries=None,
             datas=[ (path+'/img/*.png', 'img'),
                     (path+'/icons/*.png', 'icons')
                    ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='mmind_app',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='mmind_app')
app = BUNDLE(coll,
             name='mmind_app.app',
             icon='mm.icns',
             bundle_identifier=None)
