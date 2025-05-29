import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from flask import Flask, request, jsonify
import os
from pyngrok import ngrok
import uuid


# ==== CONFIG ====
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAME = "5CD-AI/Vintern-1B-v3_5"
IMAGE_SIZE = 448
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==== PREPROCESSING ====
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    processed = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in processed]
    return torch.stack(pixel_values)

# ==== LOAD MODEL ====
def load_vintern_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False
    ).eval().cuda()
    return model, tokenizer

# ==== TASK-SPECIFIC PROMPT ====
LAND_INFO_FIELDS = {
    "name": "tên của chủ đất",
    "id_no": "số giấy tờ của chủ đất",
    "land_no": "số thửa đất",
    "area": "diện tích thửa đất theo mét vuông",
    "address": "địa chỉ thửa đất",
    "license_type": "hình thức sử dụng đất",
    "license_no": "số giấy chứng nhận sử dụng đất",
    "map_no": "bản đồ số bao nhiêu"
}
ID_CARD_FIELDS = {
    "name": "họ và tên",
    "id_no": "số căn cước công dân",
    "dob": "ngày sinh",
    "gender": "giới tính",
    "nationality": "quốc tịch",
    "address": "nơi thường trú",
    "issue_date": "ngày cấp",
    "expiry_date": "có giá trị đến"
}
DRIVER_LICENSE_FIELDS = {
    "name": "họ và tên",
    "dob": "ngày sinh",
    "gender": "giới tính",
    "nationality": "quốc tịch",
    "address": "địa chỉ",
    "license_class": "hạng giấy phép",
    "license_no": "số giấy phép lái xe",
    "issue_date": "<địa danh>, ngày <dd> tháng <mm> năm <yyyy>",
    "expiry_date": "có giá trị đến"
}
# ==== DOC TYPE DETECTION ====
def detect_document_type(text: str) -> str:
    text = text.lower()
    if "giấy chứng nhận quyền sử dụng đất" in text or "thửa đất" in text or "bản đồ số" in text:
        return "land"
    elif "căn cước công dân" in text or "số cccd" in text or "nơi thường trú" in text:
        return "id_card"
    elif 'giấy phép lái xe' in text or "driver's license" in text :
      return "driver_license"
    return "unknown"

# ==== PROMPT BUILDER ====
def build_prompt_by_type(doc_type: str) -> str:
    if doc_type == "land":
        fields = LAND_INFO_FIELDS
        intro = "Dưới đây là văn bản trích xuất từ ảnh sổ đỏ.\nHãy trích xuất thông tin theo các trường sau, nếu không có thì ghi 'Không có':"
    elif doc_type == "id_card":
        fields = ID_CARD_FIELDS
        intro = "Dưới đây là văn bản trích xuất từ ảnh căn cước công dân.\nHãy trích xuất thông tin theo các trường sau, nếu không có thì ghi 'Không có':"
    elif doc_type == "driver_license":
        fields = DRIVER_LICENSE_FIELDS
        intro = "Dưới đây là văn bản trích xuất từ bằng lái xe.\nHãy trích xuất thông tin theo các trường sau, nếu không có thì ghi 'Không có':"
    else:
        return "Dưới đây là văn bản trích xuất từ ảnh.\nHãy tóm tắt nội dung văn bản dưới dạng gọn gàng."

    field_lines = [f"- {k}: {v}" for k, v in fields.items()]
    return "<image>\n" + intro + "\n" + "\n".join(field_lines) + "\n\nTrả lời dưới dạng JSON."

# ==== MODEL LOADING ====
def load_vintern_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False
    ).eval().cuda()
    return model, tokenizer

# ==== IMAGE LOADING ====

def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    resized_image = image.resize((input_size, input_size))
    return torch.stack([transform(resized_image)])

# ==== INFERENCE FUNCTION ====
def extract_info_from_image(image_path: str) -> str:
    pixel_values = load_image(image_path, input_size=IMAGE_SIZE).to(torch.bfloat16).cuda()
    model, tokenizer = load_vintern_model()

    # Run OCR with ViNTERN (text generation)
    ocr_prompt = "Dưới đây là ảnh. Hãy đọc văn bản trong ảnh và in ra toàn bộ nội dung gốc."
    ocr_response, _ = model.chat(tokenizer, pixel_values, ocr_prompt, generation_config={"max_new_tokens": 1024}, history=None, return_history=True)

    # Detect document type
    doc_type = detect_document_type(ocr_response)
    structured_prompt = build_prompt_by_type(doc_type)

    # Extract structured fields
    generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=3, repetition_penalty=2.5)
    response, _ = model.chat(tokenizer, pixel_values, structured_prompt, generation_config=generation_config, history=None, return_history=True)

    return doc_type,response

app = Flask(__name__)

@app.route('/extract', methods=['POST'])
def extract():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    file_name = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_FOLDER, file_name )
    Image.open(file.stream).convert("RGB").save(path)

    try:
        doc_type, result = extract_info_from_image(path)

        return jsonify({
            "doc_type" : doc_type,
            "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(path)


public_url = ngrok.connect(7860)
print("🔗 Public URL:", public_url)

app.run(port=7860)


