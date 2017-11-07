import ctypes


if __name__ == "__main__":

    dl = ctypes.windll.LoadLibrary(r'C:\Fussen\FlyBear1.3\bin\Win32API.dll')
    dl.Add