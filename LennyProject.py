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
        st.error(f"Error: {e}")
        return "Error", None

def main():
    st.title("Image Captioning and Audio Playing App")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        try:
            caption, audio_path = captioner(Image.open(uploaded_image))
            # Perform image captioning
            st.write("Image Caption:", caption)

            # Play audio associated with the image
            with open(audio_path, "rb") as audio_file:
                audio = audio_file.read()

            # Use st.audio to embed audio
            st.audio(audio, format="audio/mp3", start_time=0)

        except Exception as e:
            # Log the exception to identify the issue
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
