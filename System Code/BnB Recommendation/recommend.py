import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import sqlite3


# Load the necessary files
load_directory = '5001_model_save/'
scaler = joblib.load(load_directory + 'scaler.pkl')
encoder_reviewer = joblib.load(load_directory + 'encoder_reviewer.pkl')
encoder_listing = joblib.load(load_directory + 'encoder_listing.pkl')
continuous_features = joblib.load(load_directory + 'continuous_features.pkl')
categorical_features = joblib.load(load_directory + 'categorical_features.pkl')
combined_features = joblib.load(load_directory + 'combined_features.pkl')  # Load combined_features

def preprocess_data(df):
    # Standardize continuous features
    if any(feature in df.columns for feature in continuous_features):
        df[continuous_features] = scaler.transform(df[continuous_features])

    # One-hot encode categorical features (assuming the original categories are present in the dataframe)
    df = pd.get_dummies(df, columns=categorical_features)

    # Ensure df has all the columns present in combined_features
    for col in combined_features:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns based on combined_features
    df = df[combined_features + ['reviewer_id', 'listing_id', 'scores']]

    # Label encode 'reviewer_id' and 'listing_id'
    df['reviewer_id'] = encoder_reviewer.transform(df['reviewer_id'])
    df['listing_id'] = encoder_listing.transform(df['listing_id'])

    # Convert boolean type columns to float
    bool_columns = df.select_dtypes(include=[bool]).columns
    df[bool_columns] = df[bool_columns].astype(float)
    
    return df

# Load and preprocess the data
conn = sqlite3.connect('database.db')
df = pd.read_sql_query('SELECT * FROM features', conn)
df = preprocess_data(df)

# Load the model
model = load_model(load_directory + '5001_NCF_model.h5')







# ------------------------------------------------------------------------------------------
# 为老客户推荐民宿：根据用户的新输入特征作为模型的输入来提供更个性化的推荐（新输入特征也可为空）（功能1）
def recommend_homes(encoded_user_id, new_features, model, df, top_n=50):
    # 将编码后的user_id转换为原始reviewer_ID
    original_reviewer_id = encoder_reviewer.inverse_transform([encoded_user_id])[0]
    
    # 1. 获取用户历史数据
    user_data = df[df['reviewer_id'] == encoded_user_id]
    
    # 确保用户存在
    if user_data.empty:
        print(f"No historical data found for user with reviewer ID {original_reviewer_id}.")
        return None

    # 获取用户未评价的房源
    reviewed_listings = user_data['listing_id'].unique()
    potential_listings = df[~df['listing_id'].isin(reviewed_listings)].drop_duplicates(subset=['listing_id'])

    # 临时DataFrame用于设置新特征，以便我们不会改变原始数据
    temp_potential_listings = potential_listings.copy()

    # 设置新特征
    for feature, value in new_features.items():
        temp_potential_listings.loc[:, feature] = value

    # 创建模型输入
    user_ids = np.array([encoded_user_id] * len(temp_potential_listings)).astype(np.float32)
    item_ids = temp_potential_listings['listing_id'].values.astype(np.float32)
    
    # Gather the combined features
    combined_values = temp_potential_listings[combined_features].values.astype(np.float32)

    # 3. 使用之前训练的模型进行预测
    predicted_scores = model.predict([user_ids, item_ids, combined_values])
    
    # 将预测评分添加到potential_listings DataFrame
    potential_listings['predicted_scores'] = predicted_scores

    # 4. 对预测得到的评分进行排序，并选择最高分的房源进行推荐
    recommended_listings = potential_listings.sort_values(by='predicted_scores', ascending=False).head(top_n)
    
    # 5.在函数的最后，将编码后的list ID转换为原始list ID
    recommended_encoded_list_ids = recommended_listings['listing_id'].tolist()
    recommended_original_list_ids = encoder_listing.inverse_transform(recommended_encoded_list_ids).tolist()
    
    return recommended_original_list_ids




# ------------------------------------------------------------------------------------------
# 对于新用户，通过新特征找到最相似的用户，并推荐相似用户预订过的民宿和可能预订的民宿（基于用户的协同过滤）（功能2）
from sklearn.metrics.pairwise import cosine_similarity

def recommend_based_on_similarity_and_predictions(new_features, model, df, top_n=50):
    # 1. Create a DataFrame for the new user's features
    new_user_data = pd.DataFrame([new_features])
    
    # Preprocess the new user's data
    # Standardize continuous features
    if any(feature in continuous_features for feature in new_features.keys()):
        new_user_data[continuous_features] = scaler.transform(new_user_data[continuous_features])


    # Ensure all categorical features are present in new_user_data
    for feature in categorical_features:
        if feature not in new_user_data.columns:
            new_user_data[feature] = 0

    # Now you can safely one-hot encode
    new_user_data = pd.get_dummies(new_user_data, columns=categorical_features)

    # Ensure the new_user_data has all the columns present in combined_features
    for col in combined_features:
        if col not in new_user_data.columns:
            new_user_data[col] = 0
    # Convert boolean type columns to float (if any)
    bool_columns = new_user_data.select_dtypes(include=[bool]).columns
    new_user_data[bool_columns] = new_user_data[bool_columns].astype(float)


    # Select the same feature set for all users
    user_features = df[combined_features]
    
    # 2. Compute similarity between the new user and all other users
    similarities = cosine_similarity(new_user_data[combined_features], user_features)
    
    # Get the index of the most similar user
    most_similar_index = np.argmax(similarities)
    
    # Identify the most similar user to the new user
    similar_user_id = df.iloc[most_similar_index]['reviewer_id']
    
    # 3. Get the top listings booked by this user
    similar_user_data = df[df['reviewer_id'] == similar_user_id]
    top_listings_encoded = similar_user_data.sort_values(by='scores', ascending=False).head(top_n)
    
    # 4. Predict listings the similar user might like
    listings_not_seen = df[~df['listing_id'].isin(similar_user_data['listing_id'].tolist())].drop_duplicates(subset=['listing_id'])
    
    # Extract necessary data to input into the model
    user_ids = np.array([similar_user_id] * len(listings_not_seen)).astype(np.float32)
    item_ids = listings_not_seen['listing_id'].values.astype(np.float32)
    combined_values = listings_not_seen[combined_features].values.astype(np.float32)
  
    # Make predictions using the model
    predicted_scores = model.predict([user_ids, item_ids, combined_values])
    listings_not_seen['predicted_scores'] = predicted_scores
    top_predicted_listings_encoded = listings_not_seen.sort_values(by='predicted_scores', ascending=False).head(top_n)
    
    # Convert the encoded listing IDs back to the original listing IDs
    booked_homes_list_original = encoder_listing.inverse_transform(top_listings_encoded['listing_id'].tolist()).tolist()
    predicted_homes_list_original = encoder_listing.inverse_transform(top_predicted_listings_encoded['listing_id'].tolist()).tolist()
    
    # Merge the two lists ensuring that items from booked_homes_list come first
    combined_homes_list = booked_homes_list_original + predicted_homes_list_original
    
    return combined_homes_list






# # ------------------------------------------------------------------------------------------
# # 获取推荐民宿的判别函数（就是根据条件来决定使用上面2个功能函数中的某一个）
def get_recommendations(original_reviewer_id, new_features, model_path, data_path, scaler_path, encoder_reviewer_path, encoder_listing_path, continuous_features_path, categorical_features_path, combined_features_path):
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import load_model
    import joblib

    # 加载必要的文件
    scaler = joblib.load(scaler_path)
    encoder_reviewer = joblib.load(encoder_reviewer_path)
    encoder_listing = joblib.load(encoder_listing_path)
    continuous_features = joblib.load(continuous_features_path)
    categorical_features = joblib.load(categorical_features_path)
    combined_features = joblib.load(combined_features_path) 

    # 使用绝对路径加载missing_listing_ids
    #missing_ids_path = os.path.join(load_directory, 'missing_listing_ids.csv')
    #missing_ids_df = pd.read_csv(missing_ids_path)

    # 将missing_listing_ids列的值转换为一个集合
    #missing_ids = set(missing_ids_df['missing_listing_ids'].tolist())

    # 数据预处理
    df = pd.read_csv(data_path)
    df = preprocess_data(df)

    # 加载模型
    model = load_model(model_path)

    # 判断original_reviewer_id是否为0
    if original_reviewer_id != 0:
        try:
            # 根据原始的reviewer_id获取其对应的编码后的user_id
            encoded_user_id = encoder_reviewer.transform([original_reviewer_id])[0]

            # 使用编码后的user_id调用recommend_homes函数
            recommended_homes_list = recommend_homes(encoded_user_id, new_features, model, df)
        except:
            # 如果找不到用户，则为新用户推荐
            recommended_homes_list = recommend_based_on_similarity_and_predictions(new_features, model, df)
    else:
        # 使用recommend_based_on_similarity_and_predictions函数为新用户推荐
        recommended_homes_list = recommend_based_on_similarity_and_predictions(new_features, model, df)

    # 过滤掉missing_listing_ids
    #recommended_homes_list = [home_id for home_id in recommended_homes_list if home_id not in missing_ids]

    return recommended_homes_list








# ------------------------------------------------------------------------------------------
# 接口部分的转换处理函数

import pandas as pd

def frame_to_dict(df):
    # 定义期望的列
    expected_columns = set(['reviewer_id', 'host_is_superhost', 'room_type', 'geographical_location', 
                            'purpose', 'Number_of_people', 'surroundings', 'transportation'])
    
    # 使用df的列与期望的列取交集，这样无论输入的df是否包含所有列，我们都只保留期望的列
    valid_columns = list(set(df.columns) & expected_columns)
    filtered_df = df[valid_columns]
    
    dict_df = filtered_df.to_dict('list')
    
    remove_key = []
    for key, value in dict_df.items():
        value = value[0]
        dict_df[key] = value

        # 增加额外的条件，确保reviewer_id不会因为值为0而被删除
        if (value == 0 and key != 'reviewer_id') or pd.isna(value):
            remove_key.append(key)

    for key in remove_key:
        del dict_df[key]

    # 转换 'host_is_superhost' 列的值为字符串
    if 'host_is_superhost' in dict_df:
        dict_df['host_is_superhost'] = str(dict_df['host_is_superhost'])

    return dict_df


# ------------------------------------------------------------------------------------------
# 最终的推荐结果输出函数

def process_data_and_get_recommendations(feature_df):

    chatbot_5001_dic = frame_to_dict(feature_df)

    # 保存原始的 reviewer_id
    original_reviewer_id = chatbot_5001_dic['reviewer_id']

    # 从原始字典中删除 reviewer_id
    del chatbot_5001_dic['reviewer_id']

    # 创建一个映射字典将列名映射到目标格式的列名
    column_mapping = {
        'host_is_superhost': 'host_is_superhost_',
        'geographical_location': 'geographical_location_',
        'purpose': 'purpose_',
        'Number_of_people': 'Number_of_people_',  
        'surroundings': 'surroundings_',
        'transportation': 'transportation_',
        'room_type': 'room_type_'
    }

    # 使用映射字典重命名列，并构造完整的新列名
    new_features = {}
    for old_key, new_key_prefix in column_mapping.items():
        if old_key in chatbot_5001_dic:
            new_column_name = new_key_prefix + chatbot_5001_dic[old_key]
            new_features[new_column_name] = 1

    # 使用get_recommendations函数获得推荐的民宿
    recommended_original_list_ids = get_recommendations(
        original_reviewer_id, 
        new_features, 
        load_directory + '5001_NCF_model.h5', 
        load_directory + "features.csv", 
        load_directory + 'scaler.pkl', 
        load_directory + 'encoder_reviewer.pkl', 
        load_directory + 'encoder_listing.pkl', 
        load_directory + 'continuous_features.pkl', 
        load_directory + 'categorical_features.pkl', 
        load_directory + 'combined_features.pkl'
    )

    return recommended_original_list_ids

if __name__ == '__main__':
    
    feature_df = 123#输入

    if feature_df is not None:
        recommendations_listing_ids = process_data_and_get_recommendations(feature_df)



