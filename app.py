import streamlit as st
st.set_page_config(page_title='Skripsick', layout='wide')
from PIL import Image
from multiapp import MultiApp
from apps import home, data, model # import your app modules here

app = MultiApp()

st.markdown("""
# Skripsick

Multi-page app ini dibuat dengan sepenuh hati.

""")
image=Image.open('/content/skripsi/logo.png')
logo = st.columns((1, 1, 1))

#page layout


st.title('Welcome')
with logo[1]:
	st.image(image,width=200)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model", model.app)
# The main app
app.run()
