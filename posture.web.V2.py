import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime

# ================= CONFIGURATION RÉGULIÈRE =================
st.set_page_config(page_title="Analyseur Postural Pro", layout="wide")

# Utilisation du cache pour éviter de recharger le modèle et faire planter l'interface
@st.cache_resource
def load_mediapipe():
    import mediapipe as mp
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose_model = load_mediapipe()
mp_draw = mp.solutions.drawing_utils

# ================= FONCTIONS UTILES =================
def calculate_angle(p1, p2, p3):
    if not all([p1, p2, p3]): return 0.0
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0: return 0.0
    return abs(math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2))))))

def generate_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"BILAN POSTURAL : {results['nom']}", ln=True, align="C")
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Date : {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    
    # On ajoute les données texte (l'image peut être ajoutée si sauvegardée localement)
    pdf.ln(10)
    for k, v in results.items():
        if k != "nom":
            pdf.cell(0, 10, f"{k}: {v}", ln=True)
    
    filename = f"Bilan_{results['nom']}.pdf"
    pdf.output(filename)
    return filename

# ================= INTERFACE UTILISATEUR =================
st.title("🧍 Analyseur Postural Pro")

with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille = st.number_input("Taille (cm)", min_value=50, max_value=250, value=170)
    st.divider()
    source = st.radio("Source", ["📷 Caméra", "📁 Photo"])

# Initialisation des colonnes pour éviter les erreurs de "Node" (removeChild)
col_input, col_result = st.columns(2)

with col_input:
    image_data = None
    if source == "📷 Caméra":
        cam_file = st.camera_input("Prendre une photo")
        if cam_file: image_data = cam_file
    else:
        upload_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
        if upload_file: image_data = upload_file

if image_data:
    # Lecture stable de l'image
    img_pil = Image.open(image_data).convert('RGB')
    img_np = np.array(img_pil)
    
    if st.button("⚙️ LANCER L'ANALYSE", type="primary", use_container_width=True):
        with st.spinner("Analyse MediaPipe en cours..."):
            results = pose_model.process(img_np)

            if not results.pose_landmarks:
                st.error("❌ Aucun corps détecté. Reculez et montrez tout le corps.")
            else:
                # --- CALCULS ---
                lm = results.pose_landmarks.landmark
                h, w, _ = img_np.shape

                # Ratio pixels -> cm (Nez au milieu des talons)
                heel_y = (lm[29].y + lm[30].y) / 2
                pixel_h = abs(heel_y - lm[0].y) * h
                ratio = taille / pixel_h if pixel_h != 0 else 1

                # Angles et Bascules
                sh_angle = math.degrees(math.atan2((lm[11].y - lm[12].y)*h, (lm[11].x - lm[12].x)*w))
                ba_angle = math.degrees(math.atan2((lm[23].y - lm[24].y)*h, (lm[23].x - lm[24].x)*w))
                
                knee_l = calculate_angle(lm[23], lm[25], lm[27])
                knee_r = calculate_angle(lm[24], lm[26], lm[28])

                # --- DESSIN ---
                annotated_img = img_np.copy()
                mp_draw.draw_landmarks(
                    annotated_img, 
                    results.pose_landmarks, 
                    mp.solutions.pose.POSE_CONNECTIONS
                )

                # --- AFFICHAGE DES RÉSULTATS ---
                with col_result:
                    st.image(annotated_img, caption="Analyse Posturale")
                    
                    res_dict = {
                        "nom": nom,
                        "Épaules (deg)": f"{sh_angle:.1f}°",
                        "Bassin (deg)": f"{ba_angle:.1f}°",
                        "Genou G": f"{knee_l:.1f}°",
                        "Genou D": f"{knee_r:.1f}°"
                    }
                    
                    st.write("### 📊 Mesures")
                    st.table(res_dict)

                    # Génération PDF
                    pdf_path = generate_pdf(res_dict)
                    with open(pdf_path, "rb") as f:

                        st.download_button("📥 Télécharger le Bilan PDF", f, file_name=pdf_path)

