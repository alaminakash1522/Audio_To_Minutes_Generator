import os
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
from dotenv import load_dotenv



class AudioToMunite:
    def __init__(self, audio_file_path):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_id = "d:\\Models\\Mistral-7B-Instruct-v0.3"
        self.load_model()
        self.transcript = self.audio_extractor(audio_file_path)
        

    #Convert audio to text
    def audio_extractor(self, file_path):
        with open(file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    
    def load_model(self):
        # Quantization using BitsAndBytes 
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map="auto"
        )


    def format_chat(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def generate_response(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes meeting text into concise minutes."},
            {"role": "user", "content": f"Please generate the minutes from the following transcript:\n {self.transcript}"}
        ]
        input_text = self.format_chat(messages)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs.input_ids.shape[1]  # Length of the prompt
        generated_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return generated_text
    

summarizer = AudioToMunite("C:\\Users\\Alamin\\Desktop\\LLM\\LLM Practice\\Week 3\\Day 5\\Artificia_lNeurons_for_Music_and_Sound_Design_ Simon_Hutchinson.mp3")
minutes = summarizer.generate_response()
print(minutes)