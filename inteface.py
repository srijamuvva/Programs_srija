
import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import os

# Load your pre-trained model
with open('Image based Fruits or Vegtables Recognition with Calories.ipynb', 'w', encoding='utf-8'):
    #file.write(content)
    model = load_model('Image based Fruits or Vegtables Recognition with Calories.h5')

    labels = {0:'apple', 1:'banana', 2:'beetroot', 3:'bell pepper', 4:'cabbage', 5:'capsicum', 6:'carrot', 7:'cauliflower', 8:'chilli pepper', 9:'corn', 10:'cucumber', 11:'eggplant', 12:'garlic', 13:'ginger', 14:'grapes', 15:'jalepeno', 16:'kiwi', 17:'lemon', 18:'lettuce', 19:'mango', 20:'onion', 21:'orange', 22:'paprika', 23:'pear', 24:'peas', 25:'pineapple', 26:'pomegranate', 27:'potato', 28:'raddish', 29:'soy beans', 30:'spinach', 31:'sweet corn', 32:'sweet potato', 33:'tomato', 34:'turnip', 35:'watermelon'}
    fruits = ['Apple','Banana','grapes','kiwi','mango','orange','pear','pineapple','pomegranate','watermelon']
    vegetables = ['Beetroot','Bell pepper','Cabbage','Chilli Pepper','Capsicum','Cauliflower','Corn','Cucumber','Eggplant','Garlic','Ginger','Jalepeno','Lemon','Lettuce','Onion','Paprika','Peas','Potato','Raddish','Soy beans','Spinach','Sweet Corn','Sweet Potato','Turnip','Tomato']

    def fetch_calories(prediction):
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories_tag = scrap.find("div", class_="BNeawe iBp4i AP7Wnd")
        if calories_tag:
            calories = calories_tag.text
        else:
            calories = "Calories information not found"
        return calories

    def processed_img(img_path):
        img = load_img(img_path, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        print(y_class)
        y = " ".join(str(x) for x in y_class)
        y = int(y)
        res = labels[y]
        print(res)
        return res.capitalize()

    def create_upload_directory():
        upload_dir = './upload_images/'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            print(f"Directory '{upload_dir}' created successfully.")
        else:
            print(f"Directory '{upload_dir}' already exists.")

    def set_background():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
                background-attachment: fixed;
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    def welcome_page():
        st.title("Welcome to Image based Fruits or Vegtables Recognition with Calories!")
        st.write("This application helps you classify fruits or vegetables from images and provides calorie information.")
        st.write("To get started, click on the 'Start Classification' button below.")
        if st.button("Start Classification"):
            st.session_state.page = "classification"

    def classification_page():
        st.title("Image based Fruits or Vegtables Recognition with Calories")
        img_file = st.file_uploader("Choose an image", type=["jpg", "png"])
        if img_file is not None:
            img = Image.open(img_file).resize((250, 250))
            st.image(img, use_column_width=False)
            upload_dir = './upload_images/'
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            save_image_path = os.path.join(upload_dir, img_file.name)
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())
            if img_file is not None:
                result = processed_img(save_image_path)
                print(result)

                # Custom CSS for colored backgrounds
            st.markdown("""
                <style>
                .category {
                    background-color: #f0f8ff;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                .prediction {
                    background-color: #e6ffe6;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                .calories {
                    background-color: #fff0f0;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                </style>
            """, unsafe_allow_html=True)
            if result in vegetables:
                st.markdown(f'<div class="category">Category: Vegetables</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="category">Category: Fruits</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction">Predicted: {result}</div>', unsafe_allow_html=True)
            cal = fetch_calories(result)
            st.markdown(f'<div class="calories">{cal} (100 grams)</div>', unsafe_allow_html=True)
            if st.button("Finish"):
                st.session_state.page = "thank_you"

    def thank_you_page():
        st.title("Thank You for Using Image based Fruits or Vegtables Recognition with Calories!")
        st.write("We hope you found this application useful.")
        st.write("If you'd like to classify another image, please click the button below.")
        if st.button("Classify Another Image"):
            st.session_state.page = "classification"

    def main():
        create_upload_directory()
        set_background()
    
        if 'page' not in st.session_state:
            st.session_state.page = "welcome"
        if st.session_state.page == "welcome":
            welcome_page()
        elif st.session_state.page == "classification":
            classification_page()
        elif st.session_state.page == "thank_you":
            thank_you_page()
        
    if __name__ == "__main__":
        main()
