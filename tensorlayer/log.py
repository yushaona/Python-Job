import logging  
import logging.handlers 
import utils 

logfile = utils.GetApplicationDir() + "\\logs\\"
utils.ForceDirectories(logfile)
logfile = logfile + "test.log"
fileHandle = logging.handlers.TimedRotatingFileHandler(logfile,when='D',interval=1,backupCount=40) # 实例化handler   
fmt = '%(levelno)s,%(asctime)s,%(process)d,%(thread)d,%(module)s::%(funcName)s:%(lineno)s,%(message)s'  
formatter = logging.Formatter(fmt)   # 实例化formatter  
formatter.datefmt='%Y-%m-%d %H:%M:%S'
fileHandle.setFormatter(formatter)      # 为handler添加formatter  

console = logging.StreamHandler()
console.setFormatter(formatter)

logger = logging.getLogger("")

logger = logging.getLogger('tst')    # 获取名为tst的logger  
logger.addHandler(fileHandle)           # 为logger添加handler  
logger.addHandler(console) 
logger.setLevel(logging.INFO) 

def Log():
    return logger