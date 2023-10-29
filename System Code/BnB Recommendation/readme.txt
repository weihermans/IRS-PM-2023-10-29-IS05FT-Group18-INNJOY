0. 环境安装：
  1）pip install -r requirements.txt
  2）然后python -m spacy download en_core_web_lg
1. 首先打开service并运行，然后打开网页输入http://127.0.0.1:8000/index可以进行后续操作。
2. 首次点击recommend会进行注册登录，自己注册一个就可以写入db中，下次直接登录就可以。
3. Reviewer id在service.py line 50修改，现在默认为31231717。