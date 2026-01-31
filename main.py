import speech_recognition as sr
from transformers import MarianMTModel, MarianTokenizer
import os
import json
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Offline speech recognition imports
from vosk import Model, KaldiRecognizer
import pyaudio

# Network check
from ping3 import ping


# ------------------ NETWORK CHECK ------------------
def network_check():
    try:
        response = ping("8.8.8.8", timeout=2)
        return response is not None
    except Exception:
        return False


# ------------------ ONLINE SPEECH TO TEXT ------------------
def speech_to_text(language="hi-IN"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening (online)...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out")
        except sr.UnknownValueError:
            print("Sorry, I could not understand.")
        except sr.RequestError as e:
            print(f"Request error: {e}")
    return None


# ------------------ OFFLINE SPEECH TO TEXT ------------------
def offline_recognize():
    model_path = "vosk-model-en-us-0.42-gigaspeech"

    if not os.path.exists(model_path):
        print("Vosk model not found!")
        return None

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    mic = pyaudio.PyAudio()
    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4096
    )

    stream.start_stream()
    print("Listening (offline)...")

    start_time = time.time()
    text = ""

    while time.time() - start_time < 10:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print("You said:", text)
                break

    stream.stop_stream()
    stream.close()
    mic.terminate()
    return text if text else None


# ------------------ WIKIPEDIA SUMMARY ------------------
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")


def summarize_text(text):
    """
    Summarizes the given English text using BART (CNN).
    Works without transformers.pipeline().
    """

    # Tokenize input (BART max input length is 1024 tokens)
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=120,
        min_length=40,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    # Decode output
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary




# ------------------ TEXT TRANSLATION ------------------
def translate_text(text, src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return f"Translation failed: {e}"


# ------------------ MAIN MENU ------------------
if __name__ == "__main__":

    while True:
        print("\n-- Welcome to Speech Application --")
        print("1. Speech to Text")
        print("2. Summary")
        print("3. Text Translation")
        print("4. Exit")

        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Please enter a valid number.")
            continue

        if choice == 1:
            if network_check():
                print("Internet detected. Using ONLINE recognition.")
                speech_to_text()
            else:
                print("No internet. Using OFFLINE recognition.")
                offline_recognize()

        elif choice == 2:
            topic = speech_to_text()
            print("Summary:", summarize_text(topic))

        elif choice == 3:
            text = speech_to_text()
            src_lang = input("Source language code (e.g. en): ")
            tgt_lang = input("Target language code (e.g. fr): ")
            print("Translated Text:", translate_text(text, src_lang, tgt_lang))

        elif choice == 4:
            print("Exiting application...")
            break

        else:
            print("Please enter a valid option.")

