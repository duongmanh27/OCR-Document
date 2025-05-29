import os
import requests
import json
from PIL import Image



url = "https://ed81-34-87-115-68.ngrok-free.app/extract"
main_folder = "D:\\Code\\Python\\Project\\Recognition_Card_Id\\save_infor"
file_path = 'cccd_6.jpg'
document_types = {
    "id_card": [],
    "driver_license": [],
    "land": []
}

def create_folder_structure(main_folder) :
    os.makedirs(main_folder, exist_ok=True)
    for doc_type in document_types:
        doc_path = os.path.join(main_folder, doc_type)
        os.makedirs(os.path.join(doc_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(doc_path, 'texts'), exist_ok=True)
    print("✅ Đã tạo xong cấu trúc thư mục.")
def clean_json_string(json_str):
    # Loại bỏ ```json và ``` nếu có
    if json_str.startswith("```json"):
        json_str = json_str.strip("`").strip()
        lines = json_str.splitlines()
        if lines[0].startswith("json"):
            lines = lines[1:]
        if lines and lines[-1] == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return json_str.strip()

def inference():
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'image/jpeg')}
        try:
            res = requests.post(url, files=files)
        except requests.exceptions.RequestException as e:
            print("❌ Lỗi kết nối:", e)
            return None, None

    print(f"Status Code: {res.status_code}")
    print("Server raw response:", res.text)

    if res.status_code == 200:
        result = res.json()
        doc_type = result.get("doc_type", "unknown")
        raw_json = result.get("result", "").strip()


        if raw_json:
            try:
                cleaned = clean_json_string(raw_json)
                # print("Cleaned result:", cleaned)
                data = json.loads(cleaned)
                return doc_type, data
            except json.JSONDecodeError as e:
                print("❌ JSONDecodeError:", e)
        else:
            print("❌ Trường 'result' rỗng hoặc không tồn tại.")
    else:
        print("❌ Lỗi khi gửi request.")

    return None, None


def save_to_database(doc_type, data, image_path):
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        payload = {
            'name': data.get('name', ''),
            'doc_type': doc_type,
            'text_content': json.dumps(data, ensure_ascii=False)
        }

        response = requests.post("http://127.0.0.1:5000/add_document", files=files, data=payload)

        if response.status_code == 200:
            print("✅ Đã lưu vào database.")
        else:
            print("❌ Lỗi khi lưu vào database:", response.text)

def process_name_dob(name,dob) :
    name_ = name.lower().replace(" ", "_")
    dob_ =  dob.replace('/', "_").strip()
    return name_, dob_
def process_name_id_card(name,id_card) :
    name_ = name.lower().replace(" ", "_")
    id_card_ =  id_card.replace('/', "_").strip()
    return name_, id_card_
def app_run():
    create_folder_structure(main_folder)

    result = inference()
    if result is None :
        print("❌ Không thể trích xuất thông tin.")
        return
    doc_type, data = result
    save_to_database(doc_type, data, file_path)
    image = Image.open(file_path)
    path_ = os.path.join(main_folder, doc_type)
    folder_images = os.path.join(path_, 'images')
    folder_text = os.path.join(path_, 'texts')
    if doc_type == "land" :
        name, id_card = process_name_id_card(data.get('name', ''), data.get('id_no', ''))
        image_path = os.path.join(folder_images, "{}_{}.jpg".format(name, id_card))
        text_path = os.path.join(folder_text, "{}_{}.txt".format(name, id_card))
        image.save(image_path)
        with open(text_path, 'w', encoding='utf-8') as file :
            file.write(json.dumps(data, indent=2, ensure_ascii=False))
    elif doc_type == "id_card" or doc_type == 'driver_license' :
        name, dob = process_name_dob(data.get('name', ''), data.get('dob', ''))
        image_path = os.path.join(folder_images, "{}_{}.jpg".format(name, dob))
        text_path = os.path.join(folder_text,"{}_{}.txt".format(name, dob) )
        image.save(image_path)
        with open(text_path, 'w', encoding='utf-8') as file :
            file.write(json.dumps(data, indent=2, ensure_ascii=False))
    try:
        print(f"\n📄 Loại giấy tờ: {doc_type}")
        print("Kết quả nhận dạng (định dạng JSON đẹp):")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print("❌ JSON không hợp lệ sau khi làm sạch:")
        print(data)
        print(f"Lỗi: {e}")

app_run()

