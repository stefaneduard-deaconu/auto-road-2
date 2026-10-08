"""The matplotlib backend of the figure modules."""
import logging
import os

logger = logging.getLogger(__name__)


def set_backend(use_tk: bool = False) -> None:
    """Select the matplotlib backend, unless the caller already chose one via MPLBACKEND
    (tests and headless runs set MPLBACKEND=Agg)."""
    import matplotlib
    if os.environ.get('MPLBACKEND'):
        return
    if use_tk:
        matplotlib.use('TkAgg')
    else:
        try:
            matplotlib.use('module://backend_interagg')  # PyCharm's interactive backend
        except ImportError:
            logger.debug('backend_interagg is not available, keeping %s', matplotlib.get_backend())
    logger.debug('Using matplotlib backend=%s', matplotlib.get_backend())
