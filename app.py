import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from module import food_non_food, major_food_category, exact_food_category

st.set_page_config(layout="wide")

df = pd.read_json('ALL_ingredient_allergen.json')
# ingredient = pd.read_csv('final_vegetable.csv')
potential_effect_df = pd.read_csv('All_allergen_effects.csv')
receipe_df = list(df['Dish Name'])
print(receipe_df)


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match model input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize image
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Title
st.markdown("<h1 style='text-align: center;'>Food Classification and Allergen Identification</h1>", unsafe_allow_html=True)


# Model main pic
col1, col2, col3 = st.columns([4.5,6,1])

with col2:
    img = Image.open("main_pic.jpg")
    img = img.resize((400, 400))
    st.image(img)


# Create two colum left to get input and right to give output
left_column, right_column = st.columns([3,3])

# Using the left column to take the input image
with left_column:
    st.header("Upload your image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image        
        img = Image.open(uploaded_file)
        img_show = img.copy()
        img_show = img_show.resize((300, 300))
        col1, col2, col3 = st.columns([4.5,6,1])

        with col2:
            img_show = img.copy()
            img_show = img_show.resize((300, 300))
            st.image(img_show)
            # st.image(img_show, caption='Uploaded Image')

# Processing and displaying results in the right column if an image is uploaded
if uploaded_file is not None:
    with right_column:
        st.markdown("<h1 style='text-align: center;'>Result</h1>", unsafe_allow_html=True)
        
        try:
            # Resize and preprocess image
            img = img.convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0)

            # Make predictions whether it is food or not using the model
            food_not_food = food_non_food(input_tensor)
            if food_not_food == 'Non Food':
                # Define the custom CSS
                css = """
                <style>
                .centered {
                    text-align: center;
                }
                </style>
                """

                # Inject custom CSS with Markdown
                st.markdown(css, unsafe_allow_html=True)

                # Use the custom CSS class "centered" to center-align the subheader
                st.markdown("<h2 class='centered'>This is not a Food</h2>", unsafe_allow_html=True)

            else:
                # Get the predicted class in 34 classes
                major_class_predictions = major_food_category(input_tensor)
                exact_food = exact_food_category(major_class_predictions, input_tensor)

                st.subheader(f"Predicted Class:   {exact_food}")

                # Finding the dish in the dataframe
                class_name = None
                lst = []
                for curr_class in receipe_df:
                    curr_class = curr_class.title()
                    if curr_class.startswith(exact_food):
                        lst.append(curr_class)
                        y = df[df['Dish Name'] == curr_class]
                        allergen_found = list(y.iloc[:,2].values)[0]
                        st.subheader(f"Major Allergen in {curr_class}")
                        for i in allergen_found:
                            st.write(i)
                        # potential_allergen = y.iloc[:,5].values[0]


                        # find its effect and showing
                        if allergen_found[0] == 'No Allergen Present':
                            break
                        else:
                            st.subheader('Potential reaction(s)')
                            for curr_allergen in allergen_found:
                                st.write(curr_allergen)
                                curr_allergen = curr_allergen.title()

                                # Use the modified allergen name to filter the DataFrame
                                effect_row = potential_effect_df[potential_effect_df['Name'] == curr_allergen]

                                # Check if effect_row is not empty before accessing values
                                if not effect_row.empty:
                                    effect = effect_row.iloc[:,1].values[0]
                                    st.write(effect)
                                else:
                                    st.write(f"No data for {curr_allergen}")

                                print()

        except Exception as e:
            st.write("An error occurred:", e)

    for i in lst:
        if len(lst) > 1:
            ingr = df[df['Dish Name'] == i]
            ingr = ingr['Ingredients'].values[0]
            st.subheader(f"Ingredients: {i} --> {ingr}")

        else:
            ingr = df[df['Dish Name'] == i]
            ingr = ingr['Ingredients'].values[0]
            st.subheader(f"Ingredients:  {ingr}")



else:
    with right_column:
        st.write("Please upload an image to get results.")

# if __name__ == '__main__':
#     main()
