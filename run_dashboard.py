"""
Runner script for Streamlit app with Starlette GZipResponder compatibility patch.
"""
import sys
import starlette.middleware.gzip as gzip_mod

_orig_init = gzip_mod.GZipResponder.__init__

def _patched_init(self, app, minimum_size, compresslevel=9, *, thread_minimum_size=1024, **kwargs):
    return _orig_init(
        self, app, minimum_size,
        compresslevel=compresslevel,
        thread_minimum_size=thread_minimum_size,
        **kwargs
    )

gzip_mod.GZipResponder.__init__ = _patched_init

from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
    main()
