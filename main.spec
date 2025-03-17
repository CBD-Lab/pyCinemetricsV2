import sys ; 
sys.setrecursionlimit(sys.getrecursionlimit() * 5)
# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=[('E:\\dabaofinalversion\\pyCinemetricsV2-master')],
    binaries=[('E:\\dabaofinalversion\\pyCinemetricsV2-master\\algorithms\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe','ffmpeg/ffmpeg.exe'),('E:\\envs\\py310\\Lib\\site-packages\\paddle\\libs\\mklml.dll','.')],
    datas=[('algorithms','algorithms'),('ui','ui'),('models','models'),('fonts','fonts'),('E:\\envs\\py310\\Lib\\site-packages\\paddleocr\\tools','paddleocr/tools'),('E:\\envs\\py310\\Lib\\site-packages\\paddleocr\\ppocr','paddleocr/ppocr'),('E:\\envs\\py310\\Lib\\site-packages\\paddleocr\\ppstructure','paddleocr/ppstructure'),('E:\\envs\\py310\\Lib\\site-packages\\insightface\\data\\objects','insightface/data/objects')],
    hiddenimports=['paddleocr','shapely','pyclipper','argparse','imgaug','skimage','paddle','albumentations','docx','cv2','qdarktheme'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

splash = Splash('resources/splash.png',
                binaries=a.binaries, 
                datas=a.datas,
                text_pos=(20, 450),
                text_size=12,
                text_color='white')

exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pyCinemetricsV2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    icon='resources/icon.ico',
)