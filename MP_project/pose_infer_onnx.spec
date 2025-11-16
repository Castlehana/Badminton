# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['pose_infer_onnx.py'],
    pathex=[],
    binaries=[],
    datas=[('tcn.onnx', '.'), ('tcn_meta.json', '.'), ('C:\\Users\\tjddn\\Documents\\GitHub\\Badminton\\MP_project\\venv\\Lib\\site-packages\\mediapipe\\modules', 'mediapipe\\modules')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pose_infer_onnx',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
