import ftplib,socket

import time,os

#FTP数据上传和下载类
class FtpObject(object):
    def __init__(self,Ip,Port,userName,passWord):
        self.remoteIp = Ip
        self.remotePort = Port
        self.username = userName
        self.passwd = passWord
        self.ftp = None

    def uploadFile(self,remoteFileName,localFileName):
        if self.connectServer() == True:
            self.ftp.cwd(r"/ysn/")
            bufSize=2048
            fp = open(localFileName,'rb')
            self.ftp.storbinary('STOR '+remoteFileName,fp,bufSize)
            self.ftp.quit()

    def downloadFile(self,remoteFileName,localFileName):
        pass

    def connectServer(self):
        if self.ftp is None:
            print(">>正在连接FTP...<<")
            try:
                self.ftp = ftplib.FTP()
                self.ftp.connect(self.remoteIp,self.remotePort)
                self.ftp.login(self.username,self.passwd)
                print(self.ftp.getwelcome())
                return True
            except socket.error:
                print(">>远程FTP连接失败")
                return False
        return False
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ftp is not None:
            self.ftp.close()

if __name__ == "__main__":
    with  FtpObject("115.28.139.39",1900,"fussen","1q2w3e") as ftpObject:
        ftpObject.uploadFile("fileData.txt",r"D:\fileData.txt")
