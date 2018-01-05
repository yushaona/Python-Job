# coding=utf8

'''
Created on 2017年12月24日

@author: wuhaiqing
'''

import unittest
from selenium import webdriver
from time import sleep
from web_utils import page_contain
from selenium.webdriver.support.select import Select
# from selenium.expected_conditions import *
from selenium.webdriver.common.by import By

class Student(unittest.TestCase):
    
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(10)
        self.driver.maximize_window()
        self.driver.get('http://www.cjol.com/')
        
    def tearDown(self):
        sleep(2)
        self.driver.quit()
        
    def test1(self):
        self.driver.find_element_by_id("txtKeyWords_tip").clear()
        self.driver.find_element_by_id("txtKeyWords_tip").send_keys(u"自动化测试")
        
        self.driver.find_element_by_css_selector("div#locationtype_txt").click()
        sleep(1)
        
        self.driver.find_element_by_link_text("深圳").click()
        sleep(0.5)
        
        self.driver.find_element_by_xpath("//span[contains(., '福田区')]").click()
        self.driver.find_element_by_xpath("//span[contains(., '南山区')]").click()
        sleep(1)
        
        self.driver.find_element_by_css_selector("a#winLocation_dropdivselected_ok").click()
        sleep(0.5)
        
        self.driver.find_element_by_css_selector("a#btnSearch").click()
        
        all_jobs = self.driver.find_elements_by_css_selector("div#searchlist > ul")
        
        for job in all_jobs:
            degree = job.find_elements_by_css_selector("li.list_type_fifth").text
            print degree
            
            if degree == u'本科以上':
                checkbox = job.find_element_by_css_selector("li.list_type_checkbox > input")
                sleep(0.5)
                self.driver.execute_script("arguments[0].click();,checkbox")
                
        e1 = self.driver.find_element_by_css_selector("span#btnApplyJob")
        self.driver.execute_script("arguments[0].click();,e1")
        
        self.driver.find_element_by_css_selector("input#txtUserName_loginLayer").send_keys("tom")
        self.driver.find_element_by_css_selector("input#txtPassword_loginLayer_tip").click()
        self.driver.find_element_by_css_selector("input#txtPassword_loginLayer_tip").send_keys('123456')
        
#         self.driver.find_element_by_css_selector("a#btnLogin_loginLayer").click()
        
        
if __name__ == '__main__':
    unittest.main()
        
        
