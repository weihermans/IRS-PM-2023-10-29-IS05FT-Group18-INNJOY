import folium
import pandas as pd
import random
import sqlite3

def get_random_list():
    conn = sqlite3.connect('database.db')

    # 读取包含民宿信息、评分和受欢迎程度的CSV文件
    df = pd.read_sql_query('SELECT * FROM bnb_list', conn)

    final_recommendations = df

    # 计算综合推荐分数
    final_recommendations['final_score'] = final_recommendations['review_scores_rating'] + final_recommendations['popularity_score']

    # 根据综合推荐分数对民宿进行排序
    final_recommendations = final_recommendations.sort_values(by='final_score', ascending=False)
    final_recommendations.dropna(subset=['final_score'], inplace=True)

    # 输出包含所需列的结果
    selected_columns = [
        'listing_url', 'name', 'description', 'picture_url','host_url', 'host_name', 'host_is_superhost',
        'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude', 'longitude',
        'room_type', 'accommodates', 'price', 'has_availability', 'number_of_reviews',
        'review_scores_rating', 'calculated_host_listings_count', 'good_review_rate', 'popularity_score', 'final_score'
    ]

    # 随机选择每种room_type的若干民宿
    random.seed(42)  # 设置随机种子，以确保可重复性

    # 存储所有分组的推荐结果
    recommended_homestays = []

    # 遍历每种room_type
    room_types = final_recommendations['room_type'].unique()
    for room_type in room_types:
        # 筛选当前room_type的民宿
        filtered_homestays = df[df['room_type'] == room_type]

        # 随机选择若干民宿
        selected_homestays = filtered_homestays.sort_values(by='final_score', ascending=False).head(100)
        random_homestays = selected_homestays.sample(n=3)

        # 将当前分组的推荐结果添加到列表中
        recommended_homestays.append(random_homestays)

    random_recommendations = pd.concat(recommended_homestays)

    result = random_recommendations[selected_columns]
    return result

if __name__ == '__main__':
    # 创建地图对象
    m = folium.Map(location=[1.35, 103.8], zoom_start=12)  # 设置地图初始中心和缩放级别

    # 在地图上添加民宿标记
    for idx, row in get_random_list().iterrows():
        latitude = row['latitude']
        longitude = row['longitude']
        homestay_url = row['listing_url']
        image_url = row['picture_url']

        # 创建标记并添加到地图上
        marker = folium.Marker(
            location=[latitude, longitude],
            popup=f'<a href="{homestay_url}" target="_blank">{row["name"]}<br><img src="{image_url}" alt="Image" width="400"></a>',
            icon=folium.Icon(icon='cloud')
        )
        marker.add_to(m)

    m.get_root().html.add_child(folium.Element("<title>Random Exploration</title>"))
    m.save('templates/recommended_homestays_map.html')


