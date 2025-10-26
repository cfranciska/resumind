import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage 
from langchain.tools import tool
# Import modul Qdrant untuk filter
from qdrant_client.models import FieldCondition, MatchValue, Filter 

# --- Secrets ---
# Pastikan kunci-kunci ini ada di file .streamlit/secrets.toml
try:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError as e:
    st.error(f"Kunci rahasia {e} tidak ditemukan. Pastikan Anda telah mengisi file .streamlit/secrets.toml.")
    st.stop() 

# --- Inisialisasi model & DB ---
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_db = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name="resume_collection",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        # Kunci: LangChain akan mengambil teks resume dari key ini
        content_payload_key="resume_text" 
    )
except Exception as e:
    st.error(f"Gagal inisialisasi model atau koneksi database: {e}")
    st.stop()


# --- Fungsi ekstrak teks PDF ---
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# --- Tool untuk cari kandidat relevan (FINAL & FILTERED) ---
@tool
def get_relevant_docs(query: str, category_filter: str = None):
    """Cari relevan resume dari database Qdrant berdasarkan deskripsi posisi pekerjaan atau pertanyaan terkait jabatan kerja atau perbandingan antar dua posisi pekerjaan, dengan opsi category_filter. Gunakan category_filter saat Anda mengidentifikasi KATEGORI/INDUSTRI dari pertanyaan pengguna (misalnya: 'HR', 'IT', 'SALES')."""
    
    # 1. Buat Query Kontekstual
    contextual_query = f"Carikan contoh resume, CV, dan kriteria kandidat yang paling relevan untuk pertanyaan ini: {query}"
    
    # 2. Siapkan Filter Qdrant 
    qdrant_filter = None
    if category_filter and category_filter.upper() != "NONE":
        filter_value = category_filter.upper()
        
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="category", 
                    match=MatchValue(value=filter_value)
                )
            ]
        )
        
    # 3. Lakukan Search dengan Filter
    results = vector_db.similarity_search(
        query=contextual_query, 
        k=5, 
        filter=qdrant_filter
    )
    
    # 4. Format Output: LangChain sudah mengisi doc.page_content dari 'resume_text'
    docs_content = "\n---KANDIDAT:\n".join([
        f"Category: {doc.metadata.get('category', 'N/A')}\n"
        f"Resume: {doc.page_content}" 
        for doc in results
    ])
    return docs_content

# Mengikat Tool ke Model agar LLM dapat memutuskan pemanggilan
llm_with_tool = llm.bind_tools([get_relevant_docs])


# --- Fungsi pemrosesan Agen (RAG) ---
def run_rag_agent(query, uploaded_text):
    """Fungsi yang menjalankan Agent/RAG untuk menjawab pertanyaan dengan atau tanpa CV."""
    tool_messages = []
    
    # 1. Siapkan System Prompt 
    system_prompt = (
        "Anda adalah **AI ResuMind**, asisten yang sangat fokus pada analisis dan perbandingan data karier, CV, dan kriteria posisi pekerjaan. "
        "Anda memiliki satu-satunya tool eksternal yang tersedia: **'get_relevant_docs'**."
        "\n\n**ATURAN PENGGUNAAN TOOL:**"
        "\n1. **WAJIB PANGGIL TOOL** jika pertanyaan secara eksplisit terkait: **Analisis CV**, **Perbandingan Kandidat**, **Kriteria Posisi/Role**, **Mencari Kandidat Relevan**, atau **segala yang terkait dengan kepentingan pekerjaan seorang Recruiter**."
        "\n\n**EKSTRAKSI KATEGORI:** Sebelum memanggil 'get_relevant_docs', **WAJIB** analisis pertanyaan pengguna dan ekstrak kategori/industri utama (contoh: 'HR', 'IT', 'SALES'). Panggil tool dengan argumen `category_filter` yang sesuai. Jika kategori tidak teridentifikasi, set `category_filter` ke `None`."
        "\n\n**OUTPUT:** Jawablah semua pertanyaan, termasuk ringkasan dan kesimpulan, **HANYA dalam Bahasa Indonesia**."
    )
    messages = [SystemMessage(content=system_prompt)]

    # 2. Tangani input user
    if uploaded_text:
        user_message_content = f"""
        Telah diunggah CV kandidat berikut:
        --- CV KANDIDAT ---
        {uploaded_text[:2000]} 
        --- AKHIR CV ---

        Pertanyaan Anda: {query}
        Gunakan tool 'get_relevant_docs' dengan ringkasan CV ini sebagai input, untuk mencari kandidat pembanding di database.
        """
    else:
        user_message_content = query
    
    messages.append(HumanMessage(content=user_message_content))

    # --- PANGGIL LLM (STEP 1) ---
    response = llm_with_tool.invoke(messages)
    messages.append(response) 

    # 3. Cek apakah LLM memutuskan untuk memanggil Tool
    if response.tool_calls:
        
        # Eksekusi semua tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "get_relevant_docs":
                tool_output = get_relevant_docs.invoke(tool_args)
                
                # Menampilkan log pemanggilan tool
                tool_messages.append(
                    f"Tool Output (Query: {tool_args.get('query', 'N/A')[:50]}..., Filter: {tool_args.get('category_filter', 'None')}):\n"
                    f"{tool_output}" 
                )
                
                # Tambahkan ToolMessage ke history untuk panggilan ke-2
                messages.append(ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call["id"],
                    name=tool_name
                ))
            
        # --- PANGGIL LLM (STEP 2) dengan hasil tool ---
        final_response = llm.generate([messages])
        answer = final_response.generations[0][0].text
        usage = final_response.llm_output.get("token_usage", {})
        
    else:
        # LLM menjawab langsung
        final_response = llm.generate([messages])
        answer = final_response.generations[0][0].text
        usage = final_response.llm_output.get("token_usage", {})

    return answer, usage, tool_messages


# --- Streamlit UI ---
st.title("ResuMind: AI Resume Analyzer")
st.markdown("""
Chatbot cerdas untuk menganalisis CV kandidat, membandingkan pengalaman dan keterampilan dengan database, serta membantu recruiter mengambil keputusan.
""")
st.image("./Resume Agent/header_img.png") 

# --- Inisialisasi chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat history sebelumnya
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Upload CV PDF (opsional)
uploaded_file = st.file_uploader("Unggah CV kandidat (opsional, PDF)", type="pdf")
uploaded_text = None
if uploaded_file:
    uploaded_text = extract_text_from_pdf(uploaded_file)
    st.success("CV berhasil diunggah! 📄")

# Input user
query = st.text_area(
    "Tulis pertanyaan atau role yang ingin dianalisis:",
    placeholder="Contoh: Apa kriteria hard skill yang dibutuhkan seorang HR Manager?",
    height=120
)

# Tombol Kirim
if st.button("Kirim"):
    if not query.strip() and not uploaded_text:
        st.warning("Silakan tulis pertanyaan atau unggah CV untuk dianalisis.")
    elif not query.strip() and uploaded_text:
        st.warning("Apa yang ingin Anda tanyakan tentang kandidat ini?")
    else:
        # Panggil fungsi yang menangani logika tool calling
        with st.spinner("ResuMind sedang berpikir dan mencari data..."):
            try:
                answer, usage, tool_messages = run_rag_agent(query.strip(), uploaded_text)
                
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                # Perhitungan harga (Estimasi gpt-4o-mini)
                price_usd = (input_tokens * 0.15 + output_tokens * 0.6) / 1_000_000
                price_idr = 17000 * price_usd
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses permintaan: {e}")
                answer = "Maaf, terjadi kesalahan internal saat memproses permintaan Anda."
                input_tokens = 0
                output_tokens = 0
                price_idr = 0.0

        # --- Tampilkan jawaban & simpan chat history ---
        with st.chat_message("Human"):
            st.markdown(query)
            st.session_state.messages.append({"role": "Human", "content": query})

        with st.chat_message("AI"):
            st.markdown(answer)
            st.session_state.messages.append({"role": "AI", "content": answer})

        # --- Informasi tambahan ---
        with st.expander("**Tool Calls:** 🛠️"):
            if tool_messages:
                st.info("AI memanggil `get_relevant_docs` untuk mencari data dari database.")
                st.code("\n".join(tool_messages))
            else:
                st.write("Tidak ada tool yang dipanggil (Model menjawab langsung).")

        with st.expander("**History Chat (20 Pesan Terakhir):** 💬"):
            messages_history = st.session_state.messages[-20:]
            history_text = "\n".join([f'{msg["role"]}: {msg["content"][:1000]}...' for msg in messages_history]) or " "
            st.code(history_text)

        with st.expander("**Usage Details (Estimasi):** 💰"):
            st.code(
                f'Model: gpt-4o-mini\n'
                f'Input Tokens: {input_tokens}\n'
                f'Output Tokens: {output_tokens}\n'
                f'Price (Approx): Rp {price_idr:,.2f}'
            )

st.markdown("""
Tips:
- Pengguna chatbot bisa bertanya lebih dalam soal satu pekerjaan, membandingkan pekerjaan, dan atau mengunggah file dalam bentuk pdf untuk melakukan penilaian apakah kandidat itu sesuai atau tidak. 
- Jika jawaban chatbot tidak sesuai, masukan kata kunci seperti 'posisi' atau 'pekerjaan' agar chatbot mengenali bahwa pertanyaan adalah terkait dengan resume. Contoh: 'apa skill utama yang harus dimiliki chef?' Ganti pertanyaan ini menjadi 'Chef adalah pekerjaan, skill apa yang dia perlukan?'""")
