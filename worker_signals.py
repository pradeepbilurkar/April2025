from PyQt5.QtCore import QObject, pyqtSignal

class WorkerSignals(QObject):
    finished = pyqtSignal(object)  # Emit result data from worker