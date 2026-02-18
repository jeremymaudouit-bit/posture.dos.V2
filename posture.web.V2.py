import os
# ================= 0. VARIABLES D'ENV (AVANT TOUT IMPORT MEDIAPIPE) =================
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import tempfile

# ================= 1. CONFIGURATION INITIALE =================
st.set_page_config(page_title="Analyseur Postural Pro", layout="wide")

# ================= 2. INITIALISATION MEDIAPIPE (STABLE STREAMLIT CLOUD) =================
@st.cache_resource
def initialize_mediapipe():
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        return pose, mp.solutions.drawing_utils, mp.solutions.pose
    except Exception as e:
        st.error(f"Erreur d'initialisation MediaPipe : {e}")
        return None, None, None

pose_model, mp_draw, mp_pose = initialize_mediapipe()

# ================= 3. FONCTIONS UTILES =================
def calculate_angle(p1, p2, p3):
    if not all([p1, p2, p3]):
        return 0.0
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    angle = math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2))))
    return math.degrees(angle)

def generate_pdf(results, annotated_img_np=None):
    """
    Génère le PDF de synthèse.
    ✅ Ajout : si annotated_img_np est fourni, la photo est insérée PLUS PETITE dans le PDF.
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"BILAN POSTURAL : {results['nom']}", ln=True, align="C")

    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Date : {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(6)

    # ================= IMAGE DANS LE PDF (PLUS PETITE + CENTRÉE) =================
    if annotated_img_np is not None:
        # Streamlit Cloud: utiliser un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name

        # annotated_img_np est en RGB (np.array(PIL)) -> ok
        Image.fromarray(annotated_img_np).save(temp_path, quality=90)

        page_width = pdf.w - 2 * pdf.l_margin
        img_width = page_width * 0.55  # 🔽 réglage: 55% de la largeur de page (plus petit)
        x_center = (pdf.w - img_width) / 2

        pdf.image(temp_path, x=x_center, w=img_width)
        pdf.ln(10)

        try:
            os.remove(temp_path)
        except Exception:
            pass

    # ================= RESULTATS =================
    for k, v in results.items():
        if k != "nom":
            pdf.cell(0, 8, f"{k} : {v}", ln=True)

    filename = f"Bilan_{results['nom'].replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename

# ================= 4. INTERFACE =================
st.title("🧍 Analyseur Postural Pro")

with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille = st.number_input("Taille (cm)", min_value=50, max_value=250, value=170)
    st.divider()
    source = st.radio("Source", ["📷 Caméra", "📁 Photo"])

col_input, col_result = st.columns(2)

with col_input:
    image_data = None
    if source == "📷 Caméra":
        cam_file = st.camera_input("Prendre une photo")
        if cam_file:
            image_data = cam_file
    else:
        upload_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
        if upload_file:
            image_data = upload_file

# ================= 5. ANALYSE =================
if image_data:
    img_pil = Image.open(image_data).convert('RGB')
    img_np = np.array(img_pil)

    if st.button("⚙️ LANCER L'ANALYSE", type="primary", use_container_width=True):
        if pose_model is None:
            st.error("L'IA n'est pas opérationnelle.")
        else:
            with st.spinner("Analyse IA en cours..."):
                results = pose_model.process(img_np)

                if not results.pose_landmarks:
                    st.warning("⚠️ Aucun corps détecté. Assurez-vous d'être visible de la tête aux pieds.")
                else:
                    lm = results.pose_landmarks.landmark
                    h, w, _ = img_np.shape

                    sh_angle = math.degrees(math.atan2((lm[11].y - lm[12].y)*h, (lm[11].x - lm[12].x)*w))
                    ba_angle = math.degrees(math.atan2((lm[23].y - lm[24].y)*h, (lm[23].x - lm[24].x)*w))
                    knee_l = calculate_angle(lm[23], lm[25], lm[27])
                    knee_r = calculate_angle(lm[24], lm[26], lm[28])

                    annotated_img = img_np.copy()
                    mp_draw.draw_landmarks(
                        annotated_img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    with col_result:
                        st.image(annotated_img, caption="Analyse Posturale")
                        res_dict = {
                            "nom": nom,
                            "Inclinaison Épaules": f"{sh_angle:.1f}°",
                            "Inclinaison Bassin": f"{ba_angle:.1f}°",
                            "Angle Genou Gauche": f"{knee_l:.1f}°",
                            "Angle Genou Droit": f"{knee_r:.1f}°"
                        }
                        st.write("### 📊 Résultats")
                        st.table(res_dict)

                        try:
                            # ✅ on passe l'image annotée au PDF pour l'insérer en plus petit
                            path = generate_pdf(res_dict, annotated_img)
                            with open(path, "rb") as f:
                                st.download_button("📥 Télécharger le PDF", f, file_name=path)
                        except Exception as e:
                            st.error(f"Erreur PDF : {e}")
