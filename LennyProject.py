import streamlit as st
from gtts import gTTS
from PIL import Image
from transformers import pipeline
import base64

# Return caption and audiopath
get_completion = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def captioner(image):
    try:
        result = get_completion(image)
        caption = result[0]["generated_text"]

        # Use gTTS for text-to-speech
        tts = gTTS(caption, lang="en")

        # Save the audio to an MP3 file
        audio_path = "output.mp3"
        tts.save(audio_path)

        return caption, audio_path
    except Exception as e:
        # Log the exception to identify the issue
        print(f"Error: {e}")
        return "Error", None

def main():
    st.title("Image Captioning and Audio Playing App")
    uploaded_image = st.camera_input("")

    if uploaded_image is not None:

            caption, audio = captioner(Image.open(uploaded_image))
            # Perform image captioning
            st.write("Image Caption:", caption)

            # Play audio associated with the image
            audio = open("output.mp3", "rb").read()

            # Use st.markdown to embed HTML5 audio with autoplay
            st.markdown(
                f'<audio controls autoplay><source src="data:audio/mp3;base64,{base64.b64encode(audio).decode()}" type="audio/mp3"></audio>',
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
