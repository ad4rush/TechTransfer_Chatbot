# pdf_utils.py
import os
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import pytesseract

def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash-exp')

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber"""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def process_image(image_bytes, image_ext, page_num, img_index, text_content, page_content, pdf_text, model, images_folder):
    """Process image with Gemini and return description"""
    try:
        img_name = f"page_{page_num+1}_image_{img_index}.{image_ext}"
        img_path = os.path.join(images_folder, img_name)
        
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)

        prompt = (
            "Analyze this image thoroughly. Describe all visual elements, text content, "
            "and contextual significance. Be detailed but concise.\n"
            f"After OCR processing, the text in the image : {text}\n"
            f"The Text/content in this page : {text_content}\n"
            f"The Text/content before this page : {page_content}\n"
            f"The Content of the whole report : {pdf_text}."
        )

        img_pil = Image.open(BytesIO(image_bytes))
        response = model.generate_content([prompt, img_pil])
        return f"\n[Image {img_name}]\nGemini Response:\n{response.text}\n\nOCR Text:\n{text}\n"
    
    except Exception as e:
        return f"\n[Image {img_name} Error]\nDescription failed: {str(e)}\n"

def process_pdf(pdf_path, output_text_file, images_folder, api_key):
    """Main processing function"""
    os.makedirs(images_folder, exist_ok=True)
    model = configure_gemini(api_key)
    pdf_text = extract_text_from_pdf(pdf_path)
    
    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as fitz_doc, \
         open(output_text_file, "w", encoding="utf-8") as output_file:
        
        for page_num in range(len(fitz_doc)):
            pdf_page = pdf.pages[page_num]
            text_content = pdf_page.extract_text() or ""
            fitz_page = fitz_doc.load_page(page_num)
            images = fitz_page.get_images(full=True)
            image_descriptions = []
            
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                base_image = fitz_doc.extract_image(xref)
                description = process_image(
                    base_image["image"],
                    base_image["ext"],
                    page_num,
                    img_idx,
                    text_content,
                    text_content,  # Simplified for example
                    pdf_text,
                    model,
                    images_folder
                )
                image_descriptions.append(description)
            
            page_content = text_content + "\n" + "\n".join(image_descriptions)
            output_file.write(f"=== Page {page_num+1} ===\n{page_content}\n\n")
    
    return pdf_text