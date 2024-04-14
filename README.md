MediaBot ChatBot
Steps- 
1. create virtual envirnment(venv) in python
   1.1 install python virtual env library using 'pip install virtualenv'
   1.2 then create venv using 'python -m venv .venv'
   1.3 activate venv using '.\.venv\Scripts\activate'
2. download all the files and then run 'pip install -r requirements.txt' to install all the packages
3. after that create .env file and place
   GOOGLE_API_KEY="xxxxxxx"
   REPLICATE_API_TOKEN = xxxxxxxx
   add your key
4. then run 'streamlit run test_video.py'
