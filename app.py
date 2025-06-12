from IPython.display import Image
import base64
import os
from google import genai
from google.genai import types
import json
import pandas as pd
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import language_tool_python
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
    UserMessage,
    SystemMessage
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import flask
load_dotenv()



# Inisialisasi tool yang akan digunakan kembali
# Tool ini butuh koneksi internet pada saat pertama kali dijalankan
try:
    grammar_tool = language_tool_python.LanguageTool('en-US')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    print("Inisialisasi tool berhasil.")
except Exception as e:
    print(f"Gagal menginisialisasi tool, pastikan Java terinstal dan ada koneksi internet. Error: {e}")
    grammar_tool = None
    sentiment_analyzer = None

def env(variable):
    value = os.getenv(variable)
    if value is None:
        raise ValueError(f"Environment variable '{variable}' not set.")
    return value


base_prompt = """
# PERAN DAN TUJUAN
Anda adalah seorang ahli katalogisasi e-commerce dan spesialis SEO dengan pengalaman 10 tahun. Tugas Anda adalah menganalisis gambar produk dan menghasilkan metadata yang akurat, lengkap, dan dioptimalkan untuk mesin pencari.

# KONTEKS
Metadata ini akan digunakan secara langsung di platform e-commerce besar. Akurasi kategori dan tag sangat penting untuk filter pencarian, sementara deskripsi harus persuasif dan SEO-friendly untuk meningkatkan peringkat di Google dan mendorong penjualan.

# INSTRUKSI TUGAS
Berdasarkan gambar produk yang akan saya berikan, generate informasi produk dengan mengikuti semua aturan di bawah ini.

# ATURAN & BATASAN
1.  **product_name:**
    - Buat nama produk yang jelas dan deskriptif, namun tidak terlalu panjang (maksimal 70 karakter).
    - Sertakan merek jika dapat diidentifikasi.
    - Format: [Jenis Produk] [Merek (jika ada)] [Atribut Utama] - [Warna/Fitur Kunci]

2.  **tags:**
    - Buat sebuah array berisi tag yang relevan.
    - Sertakan kata kunci umum, kata kunci spesifik (long-tail), dan atribut produk.
    - Semua tag harus dalam format huruf kecil (lowercase).

3.  **category_suggestion:**
    - Berikan 1 (satu) sugesti kategori produk yang paling akurat.
    - Gunakan format breadcrumb, contoh: "Fashion Pria > Sepatu > Sneakers".

4.  **seo_description:**
    - Tulis deskripsi produk yang SEO-friendly dengan panjang 150-200 kata.
    - Deskripsi harus persuasif, menonjolkan manfaat dan fitur utama produk.
    - Secara alami, masukkan 2-3 kata kunci dari daftar tag ke dalam kalimat.
    - Akhiri dengan sebuah kalimat ajakan untuk bertindak (call-to-action).

# FORMAT OUTPUT
Format output HARUS dalam bentuk JSON yang valid tanpa teks pembuka atau penutup lainnya. Gunakan struktur berikut:
{
  "product_name": "string",
  "tags": ["string"],
  "category_suggestion": "string",
  "seo_description": "string"
}

# BAHASA output
bahasa output harus dalam bahasa inggris
"""

def generateGPTResponse(file_path):
    endpoint = "https://models.github.ai/inference"
    model = "openai/gpt-4.1"
    token = env('GITHUB_TOKEN')

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    response = client.complete(
        messages=[

            UserMessage(
                [
                        TextContentItem(text=base_prompt),
                        ImageContentItem(
                            image_url=ImageUrl.load(
                                image_file=file_path,
                                image_format="png",
                                detail=ImageDetailLevel.HIGH,
                            ),
                        ),
                    ],
            ),
        ],
        temperature=1.0,
        top_p=1.0,
        model=model
    )
    return response.choices[0].message.content


def generateGeminiResponse(file_path):
    client = genai.Client(
        api_key=env("GEMINI_API_KEY"),
    )
    with open(file_path, 'rb') as f:
      image_bytes = f.read()

    model = "gemini-2.5-flash-preview-05-20"
    response = client.models.generate_content(
    model=model,
    contents=[
      types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      ),
    base_prompt
        ]
    )
    return response.text.replace("```", '').replace("json", '')

# response_gpt = generateGPTResponse()
# response_gemini = generateGeminiResponse()
# response_gemini = response_gemini.replace("```", '').replace("json", '')
# print(response_gemini)



def _analisis_tunggal(data_json):
    """Fungsi internal untuk menganalisis satu objek JSON."""
    if not isinstance(data_json, dict):
        raise TypeError("Input harus berupa dictionary Python (hasil dari json.loads)")

    # Ekstrak data
    name = data_json.get("product_name", "")
    tags = data_json.get("tags", [])
    category = data_json.get("category_suggestion", "")
    desc = data_json.get("seo_description", "")

    # 1. Metrik Kuantitatif Dasar
    metrics = {
        "Nama (Jumlah Karakter)": len(name),
        "Nama (Jumlah Kata)": len(name.split()),
        "Jumlah Tag": len(tags),
        "Kategori (Kedalaman)": category.count('>') + 1 if category else 0,
        "Deskripsi (Jumlah Kata)": len(desc.split()),
        "Deskripsi (Jumlah Kalimat)": textstat.sentence_count(desc),
    }

    # 2. Metrik Kualitas Teks & Keterbacaan
    if desc:
        # Rata-rata kata per kalimat
        metrics["Deskripsi (Rata-rata Kata/Kalimat)"] = metrics["Deskripsi (Jumlah Kata)"] / metrics["Deskripsi (Jumlah Kalimat)"]

        # Readability (menggunakan standar en_US sebagai proxy komparatif)
        textstat.set_lang('en_US')
        metrics["Keterbacaan (Flesch Ease)"] = textstat.flesch_reading_ease(desc)
        metrics["Keterbacaan (Grade Level)"] = textstat.flesch_kincaid_grade(desc)

        # Keragaman Leksikal (TTR)
        tokens = [word.lower() for word in desc.split()]
        if len(tokens) > 0:
            unique_tokens = len(set(tokens))
            metrics["Keragaman Leksikal (TTR)"] = unique_tokens / len(tokens)
        else:
            metrics["Keragaman Leksikal (TTR)"] = 0

        # Analisis Sentimen (jika tool tersedia)
        if sentiment_analyzer:
            sentiment_score = sentiment_analyzer.polarity_scores(desc)
            metrics["Sentimen (Skor Compound)"] = sentiment_score['compound']
        else:
            metrics["Sentimen (Skor Compound)"] = "N/A"

        # Cek Gramatikal (jika tool tersedia)
        if grammar_tool:
            matches = grammar_tool.check(desc)
            metrics["Jumlah Kesalahan Gramatikal"] = len(matches)
        else:
            metrics["Jumlah Kesalahan Gramatikal"] = "N/A"

    return metrics

def analisis_komparatif_json(json1, json2, nama_model1="GPT-4.1", nama_model2="Gemini 2.5 Flash"):
    """
    Menganalisis dan membandingkan dua output JSON dari model AI.

    Args:
        json1 (dict): Output JSON pertama dalam bentuk dictionary.
        json2 (dict): Output JSON kedua dalam bentuk dictionary.
        nama_model1 (str): Nama untuk model pertama.
        nama_model2 (str): Nama untuk model kedua.

    Returns:
        pandas.DataFrame: Tabel perbandingan hasil analisis.
    """
    # Analisis masing-masing JSON
    hasil1 = _analisis_tunggal(json1)
    hasil1['Model'] = nama_model1

    hasil2 = _analisis_tunggal(json2)
    hasil2['Model'] = nama_model2

    # Metrik Komparatif
    tags1 = set(json1.get("tags", []))
    tags2 = set(json2.get("tags", []))
    tag_overlap = len(tags1.intersection(tags2))

    hasil1['Tag (Jumlah Sama)'] = tag_overlap
    hasil2['Tag (Jumlah Sama)'] = tag_overlap

    # Buat DataFrame untuk perbandingan
    df = pd.DataFrame([hasil1, hasil2])
    df = df.set_index('Model')

    # Mengatur urutan kolom agar lebih rapi
    urutan_kolom = [
        'Nama (Jumlah Karakter)', 'Nama (Jumlah Kata)', 'Jumlah Tag', 'Tag (Jumlah Sama)',
        'Kategori (Kedalaman)', 'Deskripsi (Jumlah Kata)', 'Deskripsi (Jumlah Kalimat)',
        'Deskripsi (Rata-rata Kata/Kalimat)', 'Keterbacaan (Flesch Ease)',
        'Keterbacaan (Grade Level)', 'Keragaman Leksikal (TTR)',
        'Sentimen (Skor Compound)', 'Jumlah Kesalahan Gramatikal'
    ]

    return df[urutan_kolom].T.round(2) # .T untuk transpose (membalik baris dan kolom)



app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return flask.render_template("index.html")

@app.route("/api", methods=["POST"])
def api():
    """
    Endpoint untuk menerima permintaan POST dengan gambar produk,
    menjalankan analisis, dan mengembalikan hasil dalam format JSON.
    """
    if flask.request.method == "POST":
        # Ambil file gambar dari request
        file = flask.request.files.get("file-input")
        if not file:
            return flask.jsonify({"error": "No file provided"}), 400

        # Simpan file sementara
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Jalankan analisis
        try:
            response_gpt = generateGPTResponse(file_path)
            response_gemini = generateGeminiResponse(file_path)

            data1 = json.loads(response_gpt)
            data2 = json.loads(response_gemini)

            tabel_perbandingan = analisis_komparatif_json(data1, data2)

            result_json = {
                "response_gemini": data2,
                "response_gpt": data1,
                "tabel_perbandingan": tabel_perbandingan.to_dict(),
                "base_prompt": base_prompt
            }
            return flask.jsonify(result_json), 200

        except Exception as e:
            return flask.jsonify({"error": str(e)}), 500

    return flask.jsonify({"error": "Invalid request method"}), 405

# # --- CONTOH PENGGUNAAN ---
if __name__ == "__main__":
    # Jalankan aplikasi Flask
    app.run(debug=True)


#     # 2. Konversi string JSON menjadi dictionary Python
#     data1 = json.loads(response_gpt)
#     data2 = json.loads(response_gemini)

#     # 3. Jalankan fungsi analisis
#     if grammar_tool and sentiment_analyzer:
#         tabel_perbandingan = analisis_komparatif_json(data1, data2)

#             # 4. Gabungkan kedua response ke dalam satu JSON untuk kemudahan konsumsi
#         result_json = {
#             "response_gemini": data2,
#             "response_gpt": data1,
#             "tabel_perbandingan": tabel_perbandingan.to_dict()
#         }
#         print(json.dumps(result_json, indent=2))