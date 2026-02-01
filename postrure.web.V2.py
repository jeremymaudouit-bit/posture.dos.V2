import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(page_title="Analyseur Postural Pro", layout="wide")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
# ==========================================


def calculate_angle(p1, p2, p3):
    if None in (p1, p2, p3):
        return 0.0

    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)

    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return abs(math.degrees(math.acos(dot / (mag1 * mag2))))


def generate_pdf(results, image_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"BILAN POSTURAL : {results['nom']}", ln=True, align="C")

    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Date : {datetime.now().strftime('%d/%m/%Y')}", ln=True)

    pdf.image(image_path, x=10, y=35, w=110)

    pdf.set_xy(125, 40)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Bascule")

    pdf.set_font("Arial", '', 11)
    pdf.set_xy(125, 50)
    pdf.cell(0, 8, f"Epaules : {results['ep_deg']:.1f}° ({results['ep_cm']:.1f} cm)")

    pdf.set_xy(125, 58)
    pdf.cell(0, 8, f"Bassin : {results['ba_deg']:.1f}° ({results['ba_cm']:.1f} cm)")

    pdf.set_xy(125, 75)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Angles Articulaires")

    pdf.set_font("Arial", '', 11)
    y = 85
    for label, value in [
        ("Genou gauche", results['ge_l']),
        ("Genou droit", results['ge_r']),
        ("Pied gauche", results['pi_l']),
        ("Pied droit", results['pi_r']),
    ]:
        pdf.set_xy(125, y)
        pdf.cell(0, 8, f"{label} : {value:.1f}°")
        y += 7

    filename = f"Bilan_{results['nom']}_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf.output(filename)
    return filename


# ================= UI =================

st.title("🧍 Analyseur Postural Pro – Photo & Caméra")

with st.sidebar:
    st.header("👤 Informations Patient")
    nom = st.text_input("Nom complet")
    taille = st.number_input("Taille réelle (cm)", min_value=50.0, max_value=250.0)

    st.divider()
    source = st.radio("Source image", ["📷 Caméra", "📁 Photo"])


image = None

if source == "📷 Caméra":
    cam = st.camera_input("Prenez une photo")
    if cam:
        image = Image.open(cam)

else:
    file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)


if image:
    st.image(image, caption="Image source", use_container_width=True)

    if st.button("⚙️ Lancer l'analyse", type="primary"):
        img_np = np.array(image)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            st.error("Aucun point détecté")
        else:
            lm = results.pose_landmarks.landmark
            h, w, _ = img_np.shape

            heel_y = (lm[29].y + lm[30].y) / 2
            px_h = abs(heel_y - lm[0].y) * h
            ratio = taille / px_h

            shoulder_angle = math.degrees(math.atan2(lm[11].y - lm[12].y, lm[11].x - lm[12].x))
            shoulder_cm = abs(lm[11].y - lm[12].y) * h * ratio

            hip_angle = math.degrees(math.atan2(lm[23].y - lm[24].y, lm[23].x - lm[24].x))
            hip_cm = abs(lm[23].y - lm[24].y) * h * ratio

            knee_l = calculate_angle(lm[23], lm[25], lm[27])
            knee_r = calculate_angle(lm[24], lm[26], lm[28])

            foot_l = calculate_angle(lm[25], lm[27], lm[29])
            foot_r = calculate_angle(lm[26], lm[28], lm[30])

            annotated = img_np.copy()
            mp_draw.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imwrite("analysis.jpg", annotated)

            st.session_state.results = {
                "nom": nom or "Anonyme",
                "ep_deg": shoulder_angle,
                "ep_cm": shoulder_cm,
                "ba_deg": hip_angle,
                "ba_cm": hip_cm,
                "ge_l": knee_l,
                "ge_r": knee_r,
                "pi_l": foot_l,
                "pi_r": foot_r
            }

            st.success("Analyse terminée")
            st.image(annotated, caption="Analyse annotée", use_container_width=True)

            st.text_area(
                "Résultats",
                f"""
Patient : {nom}
Taille : {taille} cm

Épaules : {shoulder_angle:.1f}° | {shoulder_cm:.1f} cm
Bassin  : {hip_angle:.1f}° | {hip_cm:.1f} cm

Genou G : {knee_l:.1f}°
Genou D : {knee_r:.1f}°
Pied G  : {foot_l:.1f}°
Pied D  : {foot_r:.1f}°
""",
                height=220
            )

            pdf_file = generate_pdf(st.session_state.results, "analysis.jpg")
            with open(pdf_file, "rb") as f:
                st.download_button("📄 Télécharger le PDF", f, file_name=pdf_file)
