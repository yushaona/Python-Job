import oss2

def GetOssBucket(bucketName):
    auth = oss2.Auth('WlZWPVisjOXliOAs', 'LsVjn2JN2PoYZqJTshTWMas20IlrX1')
    endpoint = 'http://oss-cn-qingdao.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucketName)
    return bucket

bucket = GetOssBucket('dt360')
bucket.get_object_to_file('22/47/69/27/20170529.39857810422583133.37959267354403070/1.2.826.0.1.3680043.2.461.9280450.1475417046/1.2.826.0.1.3680043.2.461.9280450.2171695171.dcm','M:/test/Logs/1.2.826.0.1.3680043.2.461.9280450.2171695171.dcm')
