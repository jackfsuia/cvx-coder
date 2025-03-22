
#下面是滚动爬取https://ask.cvxr.com/的代码
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 如果不需要界面，可以启用无头模式
# options.add_argument("--no-sandbox")
# options.add_argument("--disable-dev-shm-usage")
# options.add_experimental_option("detach", True)
service = Service("D:\chromedriver-win64\chromedriver.exe")

driver = webdriver.Chrome(service=service,options=options)
# 初始化WebDriver

# 导航到网页
driver.get('https://ask.cvxr.com/')

# 用于存储链接的列表
links = []

# 记录初始链接数量
initial_link_count = len(driver.find_elements(By.CSS_SELECTOR, 'a.title.raw-link.raw-topic-link'))
# 循环滚动并检查新链接
with open('D:\github4\web_driver\\x','w',encoding='utf-8') as f:
    while True:
        # 使用JavaScript滚动到页面底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 等待新元素出现
        WebDriverWait(driver, 20).until(lambda d: len(d.find_elements(By.CSS_SELECTOR, 'a.title.raw-link.raw-topic-link')) > initial_link_count)

        # 获取新出现的链接
        new_links = driver.find_elements(By.CSS_SELECTOR, 'a.title.raw-link.raw-topic-link')

        
        # new_links = driver.find_elements(By.TAG_NAME, 'a')
        if len(new_links) == len(links):
            break
        for link in new_links:
            if link not in links:
                links.append(link)
                f.write(link.get_attribute('href')+'\n')
                print(link.get_attribute('href'))
        # 更新初始链接数量
        initial_link_count = len(new_links)
        
        # 检查是否还有更多内容

        print(f"now links={ len(new_links)}")
# 打印所有链接
# for link in links:
#     print(link.get_attribute('href'))

# 关闭浏览器
driver.quit()








