from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd

#get data from web
def getdata(url,options,driver_path): 

    #options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(executable_path=driver_path, options=options)
    driver.get(url)
    #time.sleep(2)  # wait for the page to load
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser') 
    time.sleep(5)
    reviews = soup.find_all("span", {"data-hook": "review-body"})

    reviews = [review.text.strip() for review in reviews]
    
    driver.quit()
    return reviews

#freeze layers - not used as distilled models were highly sucsessful
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
