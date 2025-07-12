import os
import hashlib
import sys
from aiohttp import web
import logging

_has_patched = False

def verify_self_integrity():
    try:
        server_file = os.path.abspath(sys.modules['aiohttp.web'].__file__)
        server_dir = os.path.dirname(server_file)
        server_file = os.path.join(server_dir, "server.py")
        if not os.path.exists(server_file):
            try:
                import comfy.server
                server_file = os.path.abspath(comfy.server.__file__)
            except ImportError:
                logging.warning("[Apt-preset] Unable to locate server.py")
                return True
        hash_file = os.path.join(server_dir, '.server_hash')
        current_hash = hashlib.sha256(open(server_file, 'rb').read()).hexdigest()
        if not os.path.exists(hash_file):
            with open(hash_file, 'w') as f:
                f.write(current_hash)
            logging.info("[Apt-preset] server.py hash file created")
            return True
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        if current_hash != stored_hash:
            logging.critical("[Apt-preset] server.py file has been modified! Exiting.")
            sys.exit(1)
        logging.info("[Apt-preset] server.py integrity verification passed")
        return True
    except Exception as e:
        logging.error(f"[Apt-preset] File integrity verification error: {str(e)}")
        return False

def patch_server_initialization():
    global _has_patched
    if _has_patched:
        logging.info("[Apt-preset] Patch already applied, skipping.")
        return
    try:
        OriginalApplication = web.Application
        class PatchedApplication(OriginalApplication):
            def __init__(self, *args, **kwargs):
                verify_self_integrity()
                super().__init__(*args, **kwargs)
        web.Application = PatchedApplication
        _has_patched = True
        logging.info("[Apt-preset] Monkey patch successfully applied")
    except Exception as e:
        logging.error(f"[Apt-preset] Failed to apply monkey patch: {str(e)}")

patch_server_initialization()