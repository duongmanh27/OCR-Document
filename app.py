from flask import Flask, request, jsonify
import sqlite3
import os


app = Flask(__name__)
DB_PATH = 'database.db'
def init_db() :
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            doc_type TEXT,
            image_path TEXT,
            text_content TEXT
        )""")
    conn.commit()
    conn.close()

init_db()
@app.route('/add_document', methods=['POST'])
def add_document():
    name = request.form['name']
    doc_type = request.form['doc_type']
    text_content = request.form['text_content']
    image = request.files['image']

    image_path = os.path.join('uploads', image.filename)

    # Kết nối và kiểm tra xem dữ liệu đã tồn tại chưa
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM documents
        WHERE name = ? AND doc_type = ? AND text_content = ?
    ''', (name, doc_type, text_content))
    existing = cursor.fetchone()

    if existing:
        conn.close()
        return jsonify({'message': 'Document already exists, not added again'}), 409  # Conflict

    # Nếu chưa có thì lưu file và thêm vào DB
    image.save(image_path)
    cursor.execute('''
        INSERT INTO documents (name, doc_type, image_path, text_content)
        VALUES (?, ?, ?, ?)
    ''', (name, doc_type, image_path, text_content))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Document added successfully'}), 200

# API lấy tất cả dữ liệu
@app.route('/get_all_documents', methods=['GET'])
def get_all_documents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents')
    rows = cursor.fetchall()
    conn.close()

    documents = []
    for row in rows:
        documents.append({
            'id': row[0],
            'name': row[1],
            'doc_type': row[2],
            'image_path': row[3],
            'text_content': row[4]
        })

    return jsonify(documents)
@app.route('/get_document_by_info', methods=['POST'])
def get_document_by_info():
    doc_type = request.form['doc_type']
    name = request.form['name']

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if doc_type in ['driver_license', 'id_card']:
        dob = request.form.get('dob')  # ngày sinh
        cursor.execute('''
            SELECT * FROM documents 
            WHERE doc_type=? AND name=? AND text_content LIKE ?
        ''', (doc_type, name, f'%{dob}%'))

    elif doc_type == 'land':
        id_number = request.form.get('id_number')  # số căn cước
        cursor.execute('''
            SELECT * FROM documents 
            WHERE doc_type=? AND name=? AND text_content LIKE ?
        ''', (doc_type, name, f'%{id_number}%'))

    else:
        conn.close()
        return jsonify({'message': 'Loại tài liệu không hỗ trợ tìm kiếm này'}), 400

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return jsonify({'message': 'Không tìm thấy tài liệu phù hợp'}), 404

    documents = []
    for row in rows:
        documents.append({
            'id': row[0],
            'name': row[1],
            'doc_type': row[2],
            'image_path': row[3],
            'text_content': row[4]
        })

    return jsonify(documents), 200

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)