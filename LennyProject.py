import streamlit as st
from gtts import gTTS
from PIL import Image
from transformers import pipeline
import base64
import io

# Return caption and audiopath
get_completion = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def captioner(image):
    try:
        result = get_completion(image)
        caption = result[0]["generated_text"]

        # Use gTTS for text-to-speech
        tts = gTTS(caption, lang="en")

        # Save the audio to an in-memory stream
        audio_stream = io.BytesIO()
        tts.save(audio_stream)
        audio_stream.seek(0)

        return caption, audio_stream

    except Exception as e:
        # Log the exception to identify the issue
        st.error(f"Error: {e}")
        return "Error", None

def main():
    st.title("Image Captioning and Audio Playing App")
    uploaded_image = st.camera_input("Take a photo")

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            caption, audio_stream = captioner(image)
            # Perform image captioning
            st.write("Image Caption:", caption)

            # Play audio associated with the image
            audio = audio_stream.read()

            # Use st.audio to play the audio
            st.audio(audio, format="audio/mp3", start_time=0)

        except Exception as e:
            # Log the exception to identify the issue
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
