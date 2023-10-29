import pandas as pd
import sqlite3

database_name = 'database.db'

def import_csv_to_database(csv_file1, csv_file2, database_name, table_name):
    # 读取第一个CSV文件（包含所有列）
    data1 = pd.read_csv(csv_file1)

    # 读取第二个CSV文件（只包含listing_id和summary列）
    data2 = pd.read_csv(csv_file2, usecols=['listing_id', 'summary'])

    # 合并两个数据集，根据listing_id对应
    merged_data = pd.merge(data1, data2, on='listing_id', how='left')

    # 连接到数据库
    conn = sqlite3.connect(database_name)

    # 将合并后的数据存储到数据库中
    merged_data.to_sql(table_name, conn, index=False, if_exists='replace')

    # 关闭数据库连接
    conn.close()

def get_bnb_info(listing_id):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM bnb_list WHERE listing_id = ?', (listing_id,))
    result = cursor.fetchone()  # 获取查询结果的第一行记录

    conn.close()

    if result:
        columns = ['listing_id', 'listing_url', 'name', 'description', 'picture_url', 'host_id', 'host_url', 'host_name', 'host_is_superhost', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude', 'longitude', 'room_type', 'accommodates', 'price', 'has_availability', 'number_of_reviews', 'review_scores_rating', 'calculated_host_listings_count', 'good_review_rate', 'popularity_score', 'summary']
        listing_info = dict(zip(columns, result))
        return listing_info
    else:
        return {}
if __name__ == '__main__':
    import_csv_to_database('homestay_score_with_popularity.csv', 'summary_data.csv', database_name, 'bnb_list')