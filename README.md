## SECTION 1 : PROJECT TITLE
## InnJoy-A-chatbot-driven-system-for-real-time-personalized-BnB-recommendations
<img src="./img/home.png"
     style="float: left; margin-right: 0px;" />  

---
## SECTION 2: EXECUTIVE SUMMARY
InnJoy—A chatbot-driven system for real-time personalized BnB recommendations

In today's era, the applications of deep learning and large language models are not merely confined to theoretical studies. Their practical applications have been widely demonstrated across various industries. Our team is dedicated to exploring these advanced technologies' real-world applications in the online BnB booking domain. As an indispensable service in daily life, users are increasingly demanding a more personalized and efficient recommendation system for online BnB platforms. Addressing this need, after extensive research and experimentation, our team developed "InnJoy—A chatbot-driven system for real-time personalized BnB recommendations." 

Unlike traditional booking platforms, InnJoy strives to seamlessly integrate a chatbot powered by large language models with advanced neural collaborative filtering recommendation algorithms. This combination offers real-time and highly accurate BnB recommendations. Users no longer have to invest extensive time in intricate searches and filters. Instead, they interact with the chatbot, allowing InnJoy's system to swiftly and comprehensively understand users' immediate needs and emotional states, subsequently delivering the most suitable BnB choices. The system's core comprises two main components: firstly, a chatbot capable of instantaneously recognizing and deeply understanding users' needs and emotions; secondly, an efficient BnB recommendation engine based on neural collaborative filtering technology. Through real-time dialogues with users, the chatbot poses a series of precise questions to accurately capture users' preferences and requirements. In the background, based on the user's feedback, the recommendation system curates a personalized list of BnB suggestions. These recommendations are instantaneously displayed on the user's front-end interface, facilitating users to select their desired BnB and providing direct links to individual BnB websites or other major booking platforms. 

Product Features: 

1. Efficient Interaction: Through our front-end web-based chatbot, we effectively engage with users, capturing their needs and emotional states through a series of precise questions. 
2. Real-time Recommendations: In the backend, the recommendation algorithm processes the information provided by users instantly, refreshing the list of recommended BnBs, including their names, prices, and booking links. 
3. Highly Personalized: With the support of a large language model, our chatbot delves deep into users' intentions, offering more tailored questions and recommendations. 
4. Clear Business Model: We do not directly facilitate BnB bookings. Instead, we provide redirection links to major booking platforms, earning commissions from property listings, and strategically positioning ourselves to compete differently with major platforms. 
To provide you with a comprehensive understanding of InnJoy's functionality and user experience, this report includes two videos: one detailing the front-end user interaction process and another elucidating the backend algorithm. The written portion delves into the technical details and model training processes. 
We sincerely hope you'll take a deeper look at InnJoy, confident in its potential to revolutionize the online BnB booking landscape.
---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
|Tao Xu|A0285941U|1.Feature Selection<br/>2.Language model research<br/>3.Chatbot development<br/>4.Project report writing|e1221753@u.nus.edu|
|WEICHUANJIE|A0285709N|1.Ideation<br/>2.NCF model<br/>3.Project report writing<br/>4.team management|e1221521@u.nus.edu|
|Yan Zihan|A0285706W|1.Ideation<br/>2.Data Acquisition & Processing<br/>3.Random Forest Development<br/>4.Testing and Troubleshooting<br/>5.Project report writing and video making|e1221518@u.nus.edu|
|Zhang Yaoxi|A0285851U|1.User Interaction Design and Frontend Development<br/>2.Backend Development<br/>3.System Integration and Database Management<br/>4.Project Management<br/>5.Project report writing and schematic diagrams drawing|e1221663@u.nus.edu|

---
## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

`System Modelling`

[![BnB recommendation System Model](https://img.youtube.com/vi/t7P5J_ws2QU/0.jpg)](https://youtu.be/t7P5J_ws2QU "Innjoy - System Modelling")

`Use Case Demo`

[![BnB recommendation Demo](http://img.youtube.com/vi/GVtvjns7x3k/0.jpg)](https://youtu.be/GVtvjns7x3k "Innjoy - Use Case Demo")

---

## SECTION 5 : USER GUIDE

`Refer to appendix <Installation & User Guide> in project report at Github Folder: ProjectReport`

### 1.1 Install Dependencies

1. Install packages
    
        pip install -r requirements.txt

2. Install Spacy package

        python -m spacy download en_core_web_lg


### 1.2 Starting the Web Application

Run *service.py* script


        python service.py

Web is served on http://127.0.0.1:8000/index


---
## SECTION 6 : PROJECT REPORT / PAPER

`Refer to project report at Github Folder: ProjectReport`

---
