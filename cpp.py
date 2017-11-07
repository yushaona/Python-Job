
import time,calendar,os
import logging
import sys


nash={}
nash['nash1'] = '22'
nash['nash2'] = '33'

for k,v in nash.items():
    print("{}={}".format(k,v))


print(sys.argv[0][:sys.argv[0].rfind('.')]+'.log')
pyName = sys.argv[0].split('.')
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s %(asctime)s %(process)s %(thread)d %(threadName)s '
                                               '%(filename)s %(funcName)s %(message)s',
                    datefmt='%Y-%m-%d %H-%M-%S',
                    filename=sys.argv[0][:sys.argv[0].rfind('.')]+'.log',
                    filemode='a'
                    )

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)

logging.warning('This is warning log')
logging.info('This is info log')
logging.debug('This is debug log')

print(os.getcwd())
fileName = "fussen_whitelist_"+time.strftime("%Y%m%d",time.localtime(time.time()))
print(os.path.join(os.getcwd(),fileName))
print(time.strftime('%Y-%m-%d %H:%M:%S',(2017,2,3,1,1,1,1,1,1)))
print(time.localtime())

print(calendar.month(2017,9))

remoteFile = '/'.join(('/data',fileName))
print(remoteFile)
print(time.timezone)

a = (1,2,'4',3)
c = [ str(i) for i in a ]
print(c)

print('-----------------')
print([ num if num >2 else '|' for num in range(5)])

from random import  randrange
element = 0
while element < 10:
    new_element = 4 if randrange(100) > 89 else 2
    print(new_element)
    element = element +1


