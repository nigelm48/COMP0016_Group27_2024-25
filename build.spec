# build.spec
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import site
from pathlib import Path

import chromadb
chromadb_path = os.path.dirname(chromadb.__file__)  # 获取 chromadb 目录



# Get current script path
project_path = os.path.dirname(os.path.abspath(sys.argv[0]))


block_cipher = None

a = Analysis(
    ['main.py'],  # Main program file
    pathex=[project_path],  # Project path
    binaries=[],
    datas=[
        (os.path.join(project_path, 'embedding.py'), '.'),
        (os.path.join(project_path, 'populate_database.py'), '.'),
        (os.path.join(project_path, 'chroma'), 'chroma'),  # Package chroma directory
        (os.path.join(project_path, 'Qwen2.5-1.5B'), 'Qwen2.5-1.5B'),  # Package model files
        (os.path.join(project_path, 'multilingual-e5-small'), 'multilingual-e5-small'),  # Package model files
        (chromadb_path, 'chromadb')
    ],
    hiddenimports=[
        'embedding',
        'populate_database',
        # LangChain components
        'langchain_community.document_loaders',
        'langchain_community.document_loaders.pdf',
        'langchain_community.document_loaders.directory',
        'langchain_community.document_loaders.unstructured',
        'langchain_text_splitters',
        'langchain.schema.document',
        'langchain_community.vectorstores',
        'langchain_chroma',
        'langchain.prompts',
        # ChromaDB components - comprehensive imports
        'chromadb',
        'chromadb.api',
        'chromadb.api.client',
        'chromadb.api.models',
        'chromadb.api.segment',
        'chromadb.api.types',
        'chromadb.config',
        'chromadb.db',
        'hnswlib',
        'chromadb.db.impl',
        'chromadb.db.impl.sqlite',
        'chromadb.segment',
        'chromadb.segment.impl',
        'chromadb.segment.impl.metadata',
        'chromadb.segment.impl.vector',
        'chromadb.telemetry',
        'chromadb.telemetry.product',
        'chromadb.telemetry.product.posthog',
        'chromadb.utils',
        'chromadb.migrations',
        'chromadb.migrations.versions', 
        'chromadb.utils.embedding_functions',
        'chromadb.migrations.embeddings_queue',
        'chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2',
        # Unstructured components for document processing
        'unstructured',
        'unstructured.partition',
        'unstructured.documents',
        'unstructured.documents.elements',
        # Other needed packages
        'time',
        'tkinter',
        'tkinter.scrolledtext',
        'tkinter.filedialog',
        'torch',
        'transformers',
        'gc',
        'os',
        'posthog',
        'sentence_transformers',
        'sentencepiece',
        'onnxruntime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
    ],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI_RAG_Assistant',  # 可执行文件的名称
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 如果你想在运行时显示控制台窗口，设置为 True
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI_RAG_Assistant',  # 可执行文件的名称
)