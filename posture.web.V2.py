import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import mediapipe as mp

# ================= 1. CONFIGURATION (DOIT ÊTRE EN PREMIER) =================
st.set_page_config(page_title="Analyseur Postural Pro", layout="wide")

@st.cache_resource
def load_mediapipe():
    import mediapipe as mp
    import os
    import sys
    
    # --- HACK DE PERMISSION ---
    # On force Mediapipe à chercher ses modèles dans un dossier autorisé (/tmp)
    os.environ['MEDIAPIPE_BINARY_GRAPH_SUSPEND_INPUT'] = '1'
    
    try:
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0, # Gardez 0 pour la légèreté
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        st.error(f"Erreur d'initialisation : {e}")
        return None
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ================= 3. FONCTIONS UTILES =================
def calculate_angle(p1, p2, p3):
    """Calcule l'angle entre trois points."""
    if not all([p1, p2, p3]): return 0.0
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0: return 0.0
    # Calcul de l'angle en degrés
    angle = math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2))))
    return math.degrees(angle)

def generate_pdf(results):
    """Génère un rapport PDF simple."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"BILAN POSTURAL : {results['nom']}", ln=True, align="C")
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Date : {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(10)
    for k, v in results.items():
        if k != "nom":
            pdf.cell(0, 10, f"{k}: {v}", ln=True)
    
    filename = f"Bilan_{results['nom'].replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename

# ================= 4. INTERFACE UTILISATEUR =================
st.title("🧍 Analyseur Postural Pro")

with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille = st.number_input("Taille (cm)", min_value=50, max_value=250, value=170)
    st.divider()
    source = st.radio("Source", ["📷 Caméra", "📁 Photo"])

# Création des colonnes
col_input, col_result = st.columns(2)

with col_input:
    image_data = None
    if source == "📷 Caméra":
        cam_file = st.camera_input("Prendre une photo")
        if cam_file: image_data = cam_file
    else:
        upload_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
        if upload_file: image_data = upload_file

# ================= 5. TRAITEMENT DE L'ANALYSE =================
if image_data:
    img_pil = Image.open(image_data).convert('RGB')
    img_np = np.array(img_pil)
    
    if st.button("⚙️ LANCER L'ANALYSE", type="primary", use_container_width=True):
        # Vérification de sécurité
        if pose_model is None:
            st.error("L'IA n'est pas prête. Essayez de redémarrer l'application (Reboot).")
        else:
            with st.spinner("Analyse IA en cours..."):
                try:
                    # On s'assure que l'image est bien au format attendu (RGB)
                    results = pose_model.process(img_np)
                    
                    if not results or not results.pose_landmarks:
                        st.warning("⚠️ Aucun corps détecté. Assurez-vous d'être bien visible de la tête aux pieds.")
                    else:
                # --- CALCULS ---
                lm = results.pose_landmarks.landmark
                h, w, _ = img_np.shape

                # Angles et Bascules (Épaules et Bassin)
                sh_angle = math.degrees(math.atan2((lm[11].y - lm[12].y)*h, (lm[11].x - lm[12].x)*w))
                ba_angle = math.degrees(math.atan2((lm[23].y - lm[24].y)*h, (lm[23].x - lm[24].x)*w))
                
                # Angles des genoux
                knee_l = calculate_angle(lm[23], lm[25], lm[27])
                knee_r = calculate_angle(lm[24], lm[26], lm[28])

                # --- DESSIN ---
                annotated_img = img_np.copy()
                mp_draw.draw_landmarks(
                    annotated_img, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )

                # --- AFFICHAGE DES RÉSULTATS ---
                with col_result:
                    st.image(annotated_img, caption="Analyse Posturale")
                    
                    res_dict = {
                        "nom": nom,
                        "Inclinaison Épaules": f"{sh_angle:.1f}°",
                        "Inclinaison Bassin": f"{ba_angle:.1f}°",
                        "Angle Genou Gauche": f"{knee_l:.1f}°",
                        "Angle Genou Droit": f"{knee_r:.1f}°"
                    }
                    
                    st.write("### 📊 Mesures relevées")
                    st.table(res_dict)

                    # Génération et téléchargement PDF
                    try:
                        pdf_path = generate_pdf(res_dict)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="📥 Télécharger le Bilan PDF",
                                data=f,
                                file_name=pdf_path,
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.warning(f"Note : Le PDF n'a pas pu être généré ({e})")


