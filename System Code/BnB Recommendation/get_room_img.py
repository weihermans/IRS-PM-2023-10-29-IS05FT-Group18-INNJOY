import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sqlite3
import csv

database_name = 'database.db'
table_name= 'bnb_list'
# 要下载图片的Airbnb房间网址列表

# 文件保存路径
save_directory = 'room_images'

# 创建保存图片的文件夹
os.makedirs(save_directory, exist_ok=True)

def get_listing_urls(database_name, table_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    cursor.execute('SELECT listing_url FROM {}'.format(table_name))
    results = cursor.fetchall()

    conn.close()

    listing_urls = [result[0] for result in results]
    return listing_urls


def download_images_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 找到第一张房间展示的图片URL
    image_tag = soup.find('meta', property='og:image')
    if image_tag:
        image_url = image_tag['content']
        # 提取图片的文件名（使用URL中最后一个斜杠之后的部分作为临时文件名）
        temp_filename = os.path.basename(urlparse(image_url).path)
        # 下载图片
        response = requests.get(image_url)
        if response.status_code == 200:
            # 保存图片到本地文件夹，以提取的临时文件名命名
            with open(os.path.join(save_directory, temp_filename), 'wb') as f:
                f.write(response.content)
            print(f'Saved image: {temp_filename}')

            # 提取URL变量的最后一个斜杠之后的字符串作为新的文件名
            new_filename = url.split('/')[-1]+'.jpg'
            # 构造新的文件路径
            new_filepath = os.path.join(save_directory, new_filename)
            # 重命名文件
            os.rename(os.path.join(save_directory, temp_filename), new_filepath)
            print(f'Renamed image to: {new_filename}')
        else:
            print(f'Failed to download image from {url}')
    else:
        print(f'No image found on {url}')

def download_images_from_airbnb_urls(urls):
    for url in urls:
        download_images_from_url(url)

def get_listing_ids_from_database(database_name, table_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    cursor.execute('SELECT listing_id FROM {}'.format(table_name))
    results = cursor.fetchall()

    conn.close()

    listing_ids = [str(result[0]) for result in results]
    return listing_ids


def find_missing_listing_ids(image_folder, database_name, table_name, output_csv_file):
    # 获取数据库中的listing_ids
    listing_ids = get_listing_ids_from_database(database_name, table_name)

    # 获取图片文件夹中的所有文件名（带后缀）
    image_filenames = os.listdir(image_folder)

    # 从图片文件名列表中提取不带后缀的文件名
    image_names = [os.path.splitext(filename)[0] for filename in image_filenames]

    # 查找缺少的listing_id
    missing_listing_ids = [listing_id for listing_id in listing_ids if listing_id not in image_names]
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['missing_listing_ids'])
        csv_writer.writerows([[listing_id] for listing_id in missing_listing_ids])

    return missing_listing_ids


if __name__ == '__main__':
    airbnb_urls = get_listing_urls(database_name, table_name)
    print(airbnb_urls)
    #download_images_from_airbnb_urls(airbnb_urls)