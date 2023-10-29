import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
import random
import time
from nltk.corpus import wordnet
import nltk
from nltk.corpus import stopwords
from heapq import nlargest
from collections import defaultdict
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
# get the model


nlp = spacy.load('en_core_web_lg')




data_path="data/features1.csv"
df = pd.read_csv(data_path)




user_data=pd.DataFrame(columns=df.columns)
# print(user_data.columns)

# 获取 'neighbourhood_cleansed' 列中的所有不同的名称
unique_neighbourhoods = df['neighbourhood_cleansed'].unique().tolist()
unique_roomtype = df['room_type'].unique().tolist()
unique_accommodates = df['accommodates'].unique().tolist()
print(unique_accommodates)



cannot_get=['listing_id','scores','number_of_reviews','review_scores_rating']


question_neighbourhood_cleansed=['Where do you want to live?',
    'Do you have preference on the neighbourhood?',
    'Where do you want to stay?',
    "Are there specific regions or landmarks near which you'd prefer to stay?",
    "Is there a particular district or area you'd like to live in?",
    "Do you have any preferred neighborhoods or areas in mind?"
                ]

question_host_is_superhost=['Do you want to live with a superhost?',
                    "Would you prefer to stay with a host who has received high ratings?",
                    "Is it important for you to stay with a highly-rated host?",
                    "Do you want to live with a host who has received high ratings?"]

question_room_type=['What kind of room do you want to live in?',
                    'What kind of room do you want to stay in,private hotel, entire home or shared room?',
                    "Would you prefer a private space or are you open to shared accommodations?",
                    "Are you looking for a standard room, a studio, or a full apartment?",
                    "Do you have a preference for a private or shared bathroom?",
                    ]

question_price=['What is your budget?',
                      "What price range are you considering for your stay?",
                      "Do you have a maximum budget for your accommodation?"
                      ]

question_average_price=['What is your budget per night?',
                        "Is there a specific nightly rate you're aiming for?",
                        "Do you have a nightly budget in mind?",
                        ]

question_accommodates=['How large people do you want to live with?',
                    "How many individuals will be staying?",
                    "Are you looking for a space to accommodate any specific number of people?",
                    "Do you have a preference for the size of the accommodation?",
                    ]






def extract_location(text):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    defualt=random.choice(unique_neighbourhoods)
    if text is None:
        return defualt



    location_patterns = list(nlp.pipe(unique_neighbourhoods))
    matcher.add("LOCATION", location_patterns)


    doc = nlp(text)
    matches = matcher(doc)
    found_locations = [doc[start:end].text for match_id, start, end in matches]
    # 遍历文档中的命名实体
    for ent in doc.ents:
        # 如果实体的类型是地点，返回实体的文本
        if ent.label_ in ['GPE', 'LOC']:
            return ent.text
    if found_locations:
        return found_locations
    else:
        return defualt



# 定义要匹配的模式

matcher = Matcher(nlp.vocab)
def extract_price(text):
    defualt=200

    if text is None:
        return defualt
    pattern1 = [{"TEXT": {"REGEX": r"\d+(\.\d{2})?"}}, {"LOWER": "dollar"}, {"IS_PUNCT": True, "OP": "?"}]
    pattern2 = [{"TEXT": {"REGEX": r"\d+(\.\d{2})?"}}, {"LOWER": "dollars"}, {"IS_PUNCT": True, "OP": "?"}]
    pattern3 = [{"TEXT": {"REGEX": r"\d+(\.\d{2})?"}}, {"IS_PUNCT": True, "OP": "?"}]
    pattern4 = [{"TEXT": {"REGEX": r"\d+(\.\d{2})?"}}, {"LOWER": "sgd"}, {"IS_PUNCT": True, "OP": "?"}]

    # 将模式添加到Matcher
    matcher.add("PRICE", [pattern1, pattern2, pattern3, pattern4])
    doc = nlp(text)
    matches = matcher(doc)
    if matches:
        # 获取第一个匹配的价格
        match_id, start, end = matches[0]
        price = doc[start:end].text
        return price
    else:
        return defualt

def extract_average_price(text):
    defualt=100

    if text is None:
        return defualt

    pattern1 = [{"TEXT": {"REGEX": r"\d+(\.\d{2})?"}}, {"LOWER": "dollar"}, {"IS_PUNCT": True, "OP": "?"}]
    pattern2 = [{"TEXT": {"REGEX": r"\d+(\.\d{2})?"}}, {"LOWER": "dollars"}, {"IS_PUNCT": True, "OP": "?"}]
    pattern3 = [{"TEXT": {"REGEX": r"\d+(\.\d{2})?"}}, {"IS_PUNCT": True, "OP": "?"}]
    pattern4 = [{"TEXT": {"REGEX": r"\d+(\.\d{2})?"}}, {"LOWER": "sgd"}, {"IS_PUNCT": True, "OP": "?"}]

    # 将模式添加到Matcher
    matcher.add("PRICE", [pattern1, pattern2, pattern3, pattern4])
    doc = nlp(text)
    matches = matcher(doc)
    if matches:
        # 获取第一个匹配的价格
        match_id, start, end = matches[0]
        price = doc[start:end].text
        return price
    else:
        return defualt

room_size=[1,2,5,10]
def extract_room_size(text):
    defualt=2
    if text is None:
        return defualt

    # pattern1 = [{"LIKE_NUM": True}]
    pattern2 = [{"LIKE_NUM": True}, {"LOWER": "zooms"}, {"LOWER": "accommodates"}, {"LOWER": "adults"}, {"LOWER": "zoom"}]
    pattern3 = [{"LIKE_NUM": True}, {"LOWER": "people"}]

    matcher.add("ROOM_SIZE", [ pattern2, pattern3])
    doc = nlp(text)
    matches = matcher(doc)
    if matches:
        match_id, start, end = matches[0]
        return doc[start:end].text
    return defualt

def extract_room_type(text):
    default='Private room'
    if text is None:
        return default
    patterns = {
        "Private room": [
            [{"LOWER": "private"}, {"LOWER": "room"}],
            [{"LOWER": "single"}, {"LOWER": "room"}],
            [{"LOWER": "private"}]
        ],
        "Hotel room": [
            [{"LOWER": "hotel"}],
            [{"LOWER": "hotel"}, {"LOWER": "room"}],
            [{"LOWER": "hotel"}]
        ],
        "Entire home/apt": [
            [{"LOWER": "entire"}, {"LOWER": "home"}],
            [{"LOWER": "whole"}, {"LOWER": "house"}],
            [{"LOWER": "entire"}]
        ],
        "Shared room": [
            [{"LOWER": "shared"}, {"LOWER": "room"}],
            [{"LOWER": "dorm"}, {"LOWER": "room"}],
            [{"LOWER": "shared"}],
            [{"LOWER": "dorm"}],
            [{"LOWER": "with other"}],
        ]
    }

    for room_type, pattern in patterns.items():
        matcher.add(room_type, pattern)

        doc = nlp(text)
        matches = matcher(doc)
        if matches:
            match_id, start, end = matches[0]
            return nlp.vocab.strings[match_id]  # 返回匹配的房间类型
        return None


def extract_superhost(text) :
    default=1
    if text is None:
        return default

    matcher = Matcher(nlp.vocab)


    positive_patterns = [[{"LOWER": "yes"}], [{"LOWER": "prefer"}], [{"LOWER": "want"}], [{"LOWER": "superhost"}]]
    negative_patterns = [[{"LOWER": "no"}], [{"LOWER": "not"}], [{"LOWER": "don't"}],
                         [{"LOWER": "do"}, {"LOWER": "not"}]]


    matcher.add("POSITIVE", positive_patterns)
    matcher.add("NEGATIVE", negative_patterns)
    doc = nlp(text)
    matches = matcher(doc)

    positive_found = False
    negative_found = False

    for match_id, start, end in matches:
        match_label = nlp.vocab.strings[match_id]
        if match_label == "POSITIVE":
            positive_found = True
        elif match_label == "NEGATIVE":
            negative_found = True

    if positive_found and not negative_found:
        return 1  # 用户在意superhost
    elif negative_found and not positive_found:
        return 0  # 用户不在意superhost


def replace_synonyms(question):
    do_not_replace = ["do" ,'Is','is','Do']

    words = question.split()
    synonyms = []
    for word in words:
        if word.lower() in do_not_replace:
            # do not replace these words
            synonyms.append(word)
        else:
            syns = wordnet.synsets(word)
            if syns:
                lemma_names = syns[0].lemma_names()
                if lemma_names:
                    synonyms.append(random.choice(lemma_names))
                else:
                    synonyms.append(word)
            else:
                synonyms.append(word)
    return ' '.join(synonyms)


def ask_question_getInput(question_type):
    random.seed(time.time())
    if question_type == 'neighbourhood_cleansed':
        question = random.choice(question_neighbourhood_cleansed)
    elif question_type == 'host_is_superhost':
        question = random.choice(question_host_is_superhost)
    elif question_type == 'room_type':
        question = random.choice(question_room_type)
    elif question_type == 'price':
        question = random.choice(question_price)
    elif question_type == 'average_price':
        question = random.choice(question_average_price)
    elif question_type == 'accommodates':
        question = random.choice(question_accommodates)

    # question= replace_synonyms(question)


    print(question)

    return question

def exstract_feature(user_answers, type):
    if type == 'neighbourhood_cleansed':
        return {'neighbourhood_cleansed': extract_location(user_answers)}
    elif type == 'host_is_superhost':
        return {'host_is_superhost': extract_superhost(user_answers)}
    elif type == 'room_type':
        return {'room_type': extract_room_type(user_answers)}
    elif type == 'price':
        return {'price': extract_price(user_answers)}
    elif type == 'average_price':
        return {'average_price': extract_average_price(user_answers)}
    elif type == 'accommodates':
        return {'accommodates': extract_room_size(user_answers)}


#这是一个比较独立的函数，请在获取用户answer后调用
def extract_new_feature(user_answers, type='Try_potential'):
    type= type #可以使用两种type，一种是Try_most，一种是Try_potential
    new_feature = {
        'geographical_location': ['East Region', 'Central Region', 'North-East Region', 'West Region', 'North Region'],
        'purpose': ['Community Experience', 'Cultural Exploration', 'Business', 'Honeymoon', 'School',
                    'Nature Exploration', 'Shopping', 'Vacation'],
        'number_of_people': ['Family', 'All', 'Single', 'Couple', 'Party'],
        'surroundings': ['Cityscape', 'Riverside Areas', 'Hill Areas', 'Nature Reserves', 'Forests', 'Coastal Areas'],
        'transportation': ['Near MRT', 'Near Bus', 'Near Airport', 'Near Port']
    }
    old_feature = {
        'neighbourhood_cleansed': unique_neighbourhoods,
        'host_is_superhost': ['Yes', 'No'],
        'room_type': unique_roomtype,
        #如果杰哥能处理这个price的话，的less，over之类的可以调用这个，不可以就放弃
        # 'price': ['Below 50', 'over 50', 'over 100', 'less 200', 'Above 200', 'Below 100', 'Below 150', 'Below 300','Below 400','Below 500','above 400'],
        'price': ['20','30','50','80','200','120','150','300','400','500','130']
        # 'accommodates':unique_accommodates
    }


    def remove_stopwords(phrase):
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(phrase)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)

    def extract_phrases(text):
        doc = nlp(text)
        phrases = []
        for token in doc:
            if token.dep_ in ('xcomp', 'ccomp', 'advcl', 'dobj', 'attr'):
                phrase = f"{token.head.text} {token.text}"

                clean_phrase = remove_stopwords(phrase)  # 去除停用词
                if clean_phrase:  # 确保短语不为空
                    phrases.append(clean_phrase)
        # print(f"Extracted Phrases: {phrases}")
        return phrases

    def compare_similarity(phrases, text, new_feature,type):
        type=type
        similarity_list = []
        cleaned_text = remove_stopwords(text)
        cleaned_text_token = nlp(cleaned_text)
        # print(f'Cleaned Text: {cleaned_text_token}')
        # print(f'Phrases: {phrases}')
        # Compare cleaned text with feature values
        for feature_key, feature_values in new_feature.items():
            for feature_value in feature_values:
                feature_token = nlp(feature_value.lower())  # Convert to lowercase
                for word in cleaned_text_token:  # Iterate over each word in cleaned_text_token
                    similarity = word.similarity(feature_token)
                    similarity_list.append(((cleaned_text, feature_key, feature_value), similarity))

            # Compare extracted phrases with feature values
        for phrase in phrases:
            phrase_token = nlp(phrase.lower())  # Convert to lowercase
            for feature_key, feature_values in new_feature.items():
                for feature_value in feature_values:
                    feature_token = nlp(feature_value.lower())  # Convert to lowercase
                    for word in phrase_token:  # Iterate over each word in phrase_token
                        similarity = word.similarity(feature_token)
                        similarity_list.append(((phrase, feature_key, feature_value), similarity))

        # Select the top 3 most similar items

        sorted_similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)
        if type == "Try_most":
            # Collect the top 10 most similar items with unique Feature Keys
            top_10_dict = {}  # Create an empty dictionary to hold the results
            seen_keys = set()
            for pair, sim in sorted_similarity_list:
                phrase, feature_key, feature_value = pair
                if sim >= 0.3 and feature_key not in seen_keys and len(top_10_dict) < 10:
                    seen_keys.add(feature_key)
                    top_10_dict[feature_key] = feature_value  # Add the feature_key and feature_value to the dictionary

            #     # Print the top 10 pairs for debugging purposes
            # for feature_key, feature_value in top_10_dict.items():
            #     print(f'Feature Key: {feature_key}, Feature Value: {feature_value}')

            return top_10_dict  # Return the dictionary containing the top 10 pairs

        elif type == "Try_potential":
            # Select the top 3 most similar items
            top_pair = nlargest(3, similarity_list, key=lambda x: x[1])

            # for pair, sim in top_pair:#留下打印窗口
            #     phrase, feature_key, feature_value = pair
            #     print(
            #         f'Phrase: {phrase}, Feature Key: {feature_key}, Feature Value: {feature_value}, Similarity: {sim}')


            if top_pair and top_pair[0][1] > 0.3:
                pair, sim = top_pair[0]
                phrase, feature_key, feature_value = pair
                # print(f'Phrase: {phrase}, Feature Key: {feature_key}, Feature Value: {feature_value}, Similarity: {sim}')
                return {feature_key: feature_value} # 返回最相似的特征和键
            else:
                # print("No match found with similarity greater than 0.4")
                return None  # 如果没有找到相似度大于0.4的匹配项，则返回None



    # 函数内部调用函数
    phrases = extract_phrases(user_answers)
    if type == "Try_most":
        similarity_list = compare_similarity(phrases, user_answers, {**new_feature,**old_feature},type)
    elif type == "Try_potential":
        similarity_list = compare_similarity(phrases, user_answers, new_feature, type)

    return similarity_list




def get_feature(*args):
    #请填写，'neighbourhood_cleansed', 'host_is_superhost', 'room_type', 'price', 'average_price', 'accommodates'中一个或者多个
    user_answers = {}
    columns = ['listing_id', 'reviewer_id', 'scores', 'host_is_superhost',
               'neighbourhood_cleansed', 'room_type', 'price', 'number_of_reviews',
               'review_scores_rating', 'calculated_host_listings_count',
               'good_review_rate', 'accommodates', 'average_price']

    extract_functions= {'neighbourhood_cleansed': extract_location,
                        'host_is_superhost': extract_superhost,
                        'room_type': extract_room_type,
                        'price': extract_price,
                        'average_price': extract_average_price,
                        'accommodates': extract_room_size}

    user_feature= pd.DataFrame(columns=columns)
    user_answers= {key: func(None) for key, func in extract_functions.items()}


    # 遍历每个传入的参数（即问题类型）
    for question_type in args:
        # 确保 question_type 是一个字符串
        if isinstance(question_type, str):
            # 调用 ask_question_getInput 函数并存储用户的回答
            user_answer = ask_question_getInput(question_type) if question_type in args else None
            extract_function = extract_functions.get(question_type)
            if extract_function is not None:
                user_answers[question_type] = extract_function(user_answer)
        else:
            print(f"Invalid question type: {question_type}. It should be a string.")

    new_row = {col: 0 for col in columns}  # 初始化所有列的值为0
    for key, value in user_answers.items():
        # 根据用户的回答更新新行的字典
        if key in columns:
            new_row[key] = value

    # 将新行添加到DataFrame
    user_feature.loc[len(df)] = new_row


    # 转换格式，方便杰哥用
    dtype_dict = {
        'listing_id': 'int64',
        'reviewer_id': 'int64',
        'scores': 'int64',
        'host_is_superhost': 'object',
        'neighbourhood_cleansed': 'object',
        'room_type': 'object',
        'price': 'int64',
        'number_of_reviews': 'int64',
        'review_scores_rating': 'float64',
        'calculated_host_listings_count': 'int64',
        'good_review_rate': 'float64',
        'accommodates': 'int64',
        'average_price': 'float64'
    }
    user_feature = user_feature.astype(dtype_dict)
    user_feature = user_feature.reset_index(drop=True)

    return user_feature

def generate_userFrame(reviewer_id):
    """
        This function generates a user frame based on the reviewer's id.

        Parameters:
        reviewer_id (int): The id of the reviewer. This should be an integer.

        Returns:
        dict: A dictionary representing the user frame.
        """
    cannot_get = ['listing_id', 'scores', 'number_of_reviews', 'review_scores_rating','good_review_rate','calculated_host_listings_count']
    dtype_dict = {
        'listing_id': 'Int64',  # Use 'Int64' instead of 'int64'
        'reviewer_id': 'Int64',
        'scores': 'Int64',
        'host_is_superhost': 'object',
        'neighbourhood_cleansed': 'object',
        'room_type': 'object',
        'price': 'Int64',  # Use 'Int64' instead of 'int64',可以在obeject 换，看杰哥怎么处理less之类的字符串
        'number_of_reviews': 'Int64',
        'review_scores_rating': 'float64',
        'calculated_host_listings_count': 'Int64',
        'good_review_rate': 'float64',
        'accommodates': 'Int64',
        'average_price': 'float64',
        'geographical_location': 'object',
        'purpose': 'object',
        'number_of_people': 'object',
        'surroundings': 'object',
        'transportation': 'object'
    }
    default_values = {
        'listing_id': 0,
        'reviewer_id': reviewer_id,
        'scores': 0,
        'host_is_superhost': np.nan,  # Use np.nan instead of None
        'neighbourhood_cleansed': np.nan,
        'room_type': np.nan,
        'price': np.nan,
        'number_of_reviews': 0,
        'review_scores_rating': 0,
        'calculated_host_listings_count': 0,
        'good_review_rate':0,
        'accommodates': np.nan,
        'average_price': np.nan,
        'geographical_location': np.nan,
        'purpose': np.nan,
        'number_of_people': np.nan,
        'surroundings': np.nan,
        'transportation': np.nan
    }

    user_frame_df = pd.DataFrame([default_values])
    user_frame_df = user_frame_df.astype(dtype_dict)
    return user_frame_df

def update_userFrame(userFrame, *args):
    dtype_dict = {
        'listing_id': 'Int64',
        'reviewer_id': 'Int64',
        'scores': 'Int64',
        'host_is_superhost': 'object',
        'neighbourhood_cleansed': 'object',
        'room_type': 'object',
        'price': 'Int64',
        'number_of_reviews': 'Int64',
        'review_scores_rating': 'float64',
        'calculated_host_listings_count': 'Int64',
        'good_review_rate': 'float64',
        'accommodates': 'Int64',
        'average_price': 'float64',
        'geographical_location': 'object',
        'purpose': 'object',
        'number_of_people': 'object',
        'surroundings': 'object',
        'transportation': 'object'
    }
    updated_userFrame = userFrame.copy()

    #这行代码后续看看
    def convert_to_int(value):
        # 如果 value 是字符串并且可以转换为数字，则转换为整数
        if isinstance(value, str) and value.isnumeric():
            return int(value)
        return value

    for arg in args:
        if arg is not None and isinstance(arg, dict):
            for key, value in arg.items():
                if key in updated_userFrame.columns and value is not None:
                    value = convert_to_int(value) if key == 'price' else value
                    updated_userFrame[key] = updated_userFrame[key].where(updated_userFrame[key].notna(), value)

    updated_userFrame = updated_userFrame.astype(dtype_dict, errors='ignore')

    none_columns = updated_userFrame.columns[updated_userFrame.isna().any()].tolist()
    potential_features = ['geographical_location', 'purpose', 'number_of_people', 'surroundings', 'transportation']
    none_columns = [f for f in none_columns if f not in potential_features]

    return updated_userFrame, none_columns



if __name__=="__main__":
    # 请填写，'neighbourhood_cleansed', 'host_is_superhost', 'room_type', 'price', 'average_price', 'accommodates'中一个或者多个
    pd.options.display.max_columns = None



    #使用方法，第一步生成一个新用户的frame,请新用户请输入用户id，老用户请从数据库中调一个

    user_f=generate_userFrame(555555)
    print(user_f)

    #第二步，获取用户回答，可以genral的提示
    print('Please generally describe your trip and any room you want:')
    # answer=input()
            #此处以这个例子代替
    answer="We are planning to spend a delightful weekend in Singapore. We'd prefer to stay in the Central Region as there are many shopping malls and restaurants there. We are a large family, so we need a place that's suitable for children and the elderly. We enjoy nature, so we'd also like to take walks in nearby Nature Reserves or Riverside Areas. We prefer to stay close to an MRT station, so it's easy for us to explore the city. The main purposes of our trip are shopping and cultural exploration. Lastly, we hope to find accommodation that is reasonably priced, clean, and comfortable."

    #第三步，获取用户的特征，调用extract_new_feature函数，返回一个字典，然后更新用户的frame
        # 注意这个函数有两个模式，一个是Try_most，一个是Try_potential，Try_most是返回最相似的10个，Try_potential是返回最相似的一个
        #Try most会尝试获取尽可能多的特征，Try_potential只会获取 'geographical_location''purpose' 'number_of_people''surroundings''transportation'这几个隐形特征
        #所以请在第一次使用时使用Try_most，然后在之后几次次使用时使用Try_potential

    user_feature=extract_new_feature(answer,"Try_most")
    print(user_feature)

    #第四步，更新用户的frame,请使用update_userFrame函数，返回更新后的用户frame和缺失的特征
        #这个函数返回两个值，一个更新后的frame,一个是缺失的特征，如果没有缺失的特征，可以打印这些缺失的特征，然后让用户补充
        #'geographical_location''purpose' 'number_of_people''surroundings''transportation'这几个特征不会强制获得，所以可能出现一直为none的情况
    user_f,none_columns = update_userFrame(user_f,user_feature)
    print(user_f)
    print(none_columns)

    #第五步，调用以前的函数回答问题，获取用户的特征
    #假设accommodates这个特征没有拿到，就继续提问,同时注意调用extract_new_feature函数时，使用Try_potential模式
    #然后同时update_userFrame函数，更新用户的frame
    #能使用这个方法询问的只有'neighbourhood_cleansed', 'host_is_superhost', 'room_type', 'price', 'average_price', 'accommodates'中一个或者多个

    # question,answer=ask_question_getInput('accommodates')
    #假设问答为
    question="How many people are you traveling with?"
    answer="We are a large family, so we need a place that's suitable for children and the elderly."


    user_feature1=exstract_feature(answer,'accommodates')
    user_feature2=extract_new_feature(answer,"Try_potential")

    user_f,none_columns = update_userFrame(user_f,user_feature1,user_feature2)
    print(user_f)
    print(none_columns)









