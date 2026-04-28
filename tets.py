from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image

processor=TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model=VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

image=Image.open("com.jpeg").convert("RGB")

pixel_values=processor(image,return_tensors="pt").pixel_values
generated_ids=model.generate(pixel_values)
generated_text=processor.batch_decode(generated_ids,skip_special_tokens=True)
print(generated_text)