import sys
import streamlit.web.server.starlette.starlette_gzip_middleware as gzip_mw

# Patch Streamlit's _MediaAwareGZipResponder for Starlette / Python 3.14 compatibility
gzip_mw._MediaAwareGZipResponder.__init__ = lambda self, app, minimum_size=500, compresslevel=9, **kwargs: super(gzip_mw._MediaAwareGZipResponder, self).__init__(app, minimum_size=minimum_size, compresslevel=compresslevel, thread_minimum_size=1024)

from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.port", "8502"]
    main()
