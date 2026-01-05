import streamlit as st
from PIL import Image
import keras
import sys
import os
import numpy as np
import base64
from streamlit_drawable_canvas import st_canvas

from utils.inference import predict_mnist
from utils.style import apply_style
from training.utils.model_definition import SimpleCNN_MNIST

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction - MNIST CNN",
    page_icon="🎯",
    layout="wide"
)

# Appliquer le style global (Poppins)
apply_style()

# CSS personnalisé - Design professionnel moderne
st.markdown("""
<style>
    /* Palette de couleurs professionnelle */
    :root {
        --primary-color: #1e3a8a;
        --accent-color: #3b82f6;
        --text-dark: #1f2937;
        --text-light: #6b7280;
        --success-color: #059669;
        --white: #ffffff;
    }

    /* Background avec dégradé subtil */
    .main .block-container {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }

    /* Animations modernes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.9;
            transform: scale(1.02);
        }
    }

    /* Avatar arrondi */
    .author-avatar-small {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        border: 2px solid var(--accent-color);
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        display: inline-block;
        vertical-align: middle;
        margin-right: 12px;
        transition: transform 0.3s ease;
    }

    .author-avatar-small:hover {
        transform: scale(1.1);
    }

    /* Titre principal avec animation */
    .main-title {
        color: var(--primary-color);
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.6s ease;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: var(--text-light);
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease;
    }

    .author-name {
        color: var(--primary-color);
        font-weight: 600;
    }

    /* Carte de résultat principal avec glassmorphism */
    .result-card {
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.95) 0%, rgba(219, 234, 254, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 2px solid var(--accent-color);
        margin: 1rem 0 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        animation: fadeInUp 0.5s ease;
        transition: all 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
    }

    .result-label {
        color: var(--text-light);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }

    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.25rem 0;
        line-height: 1;
        animation: pulse 2s ease-in-out infinite;
    }

    .confidence-label {
        color: var(--success-color);
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* Séparateur */
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #e5e7eb, transparent);
        margin: 2rem 0 1.5rem 0;
    }

    /* Top 3 header */
    .top3-header {
        color: var(--primary-color);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Quality score card avec glassmorphism */
    .quality-card {
        background: rgba(249, 250, 251, 0.9);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(229, 231, 235, 0.6);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        animation: fadeInUp 0.7s ease;
        transition: all 0.3s ease;
    }

    .quality-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: var(--accent-color);
    }

    .quality-header {
        color: var(--primary-color);
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .quality-score {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .quality-score-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }

    .quality-level {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .quality-level.excellente {
        background: #d1fae5;
        color: #065f46;
    }

    .quality-level.bonne {
        background: #dbeafe;
        color: #1e40af;
    }

    .quality-level.moyenne {
        background: #fef3c7;
        color: #92400e;
    }

    .quality-level.faible {
        background: #fee2e2;
        color: #991b1b;
    }

    .quality-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.5rem;
        font-size: 0.75rem;
    }

    .quality-metric {
        background: white;
        padding: 0.5rem;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
    }

    .quality-metric-label {
        color: var(--text-light);
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .quality-metric-value {
        color: var(--text-dark);
        font-weight: 700;
    }

    /* Section header */
    .section-header {
        color: var(--primary-color);
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
    }

    /* Image container */
    .image-container {
        background: var(--white);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: var(--text-light);
        font-size: 0.9rem;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }

    /* Ajustements pour les boutons radio */
    .stRadio > label {
        font-weight: 600;
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# Charger le modèle CNN
@st.cache_resource
def load_model():
    # Chemin depuis la racine du projet (où se trouve streamlit_app.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'mnist_cnn.keras')
    
    # Debug : afficher le chemin (à supprimer après)
    print(f"🔍 Chemin du modèle : {model_path}")
    print(f"📁 Existe ? {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle introuvable : {model_path}")
        st.stop()
    
    return keras.models.load_model(model_path)

# Charger le dataset MNIST
@st.cache_data
def load_mnist_dataset():
    """Charge le dataset MNIST pour le mode test"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (x_test, y_test)

model = load_model()

# Fonction helper pour afficher le score de qualité
def display_quality_score(quality_score):
    """Affiche le score de qualité du preprocessing de manière visuelle"""
    if quality_score is None:
        return

    level_class = quality_score['quality_level'].lower()

    st.markdown(f"""
    <div class="quality-card">
        <div class="quality-header">📊 Qualité du Preprocessing</div>
        <div class="quality-score">
            <span class="quality-score-value">{quality_score['global_score']}/1.0</span>
            <span class="quality-level {level_class}">{quality_score['quality_level']}</span>
        </div>
        <div class="quality-metrics">
            <div class="quality-metric">
                <div class="quality-metric-label">Contraste</div>
                <div class="quality-metric-value">{quality_score['contrast']:.1f} ({quality_score['contrast_score']:.0%})</div>
            </div>
            <div class="quality-metric">
                <div class="quality-metric-label">Taille</div>
                <div class="quality-metric-value">{quality_score['size']}px ({quality_score['size_score']:.0%})</div>
            </div>
            <div class="quality-metric">
                <div class="quality-metric-label">Aspect ratio</div>
                <div class="quality-metric-value">{quality_score['aspect_ratio']:.2f} ({quality_score['aspect_score']:.0%})</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# En-tête avec avatar
avatar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'profile.jpg')
avatar_html = ""
if os.path.exists(avatar_path):
    with open(avatar_path, "rb") as f:
        avatar_data = base64.b64encode(f.read()).decode()
        avatar_html = f'<img src="data:image/jpeg;base64,{avatar_data}" class="author-avatar-small" />'

st.markdown('<h1 class="main-title">🎯 Reconnaissance de chiffres manuscrits</h1>', unsafe_allow_html=True)

# Instructions (seul expander conservé)
with st.expander("ℹ️ Comment utiliser cette application", expanded=False):
    st.markdown("""
    **Modes disponibles**
    - **🖼️ Upload** : Téléchargez une image de chiffre manuscrit (PNG, JPG, JPEG)
    - **📸 Caméra** : Prenez une photo d'un chiffre écrit sur papier
    - **✍️ Dessiner** : Dessinez un chiffre directement sur le canvas""")

st.markdown("<br>", unsafe_allow_html=True)

# Choix du mode
mode = st.radio(
    "**Choisissez un mode**",
    ["🖼️ Upload", "📸 Caméra", "✍️ Dessiner"],
    horizontal=True
)

# Variables pour rembg et TTA (valeurs par défaut)
rembg_model = "u2netp"
use_tta = False

st.markdown("<br>", unsafe_allow_html=True)

if mode == "🖼️ Upload":
    uploaded_file = st.file_uploader("Choisir une image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-header">Image originale</div>', unsafe_allow_html=True)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image := Image.open(uploaded_file), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">Résultats de l\'analyse</div>', unsafe_allow_html=True)

            with st.spinner("🔍 Analyse en cours..."):
                top3, steps, quality_score = predict_mnist(
                    image, model,
                    return_steps=True,
                    rembg_model=rembg_model,
                    use_tta=use_tta,
                    return_quality=True
                )

            # Résultat principal
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prédiction</div>
                <div class="prediction-value">{top3[0][0]}</div>
                <div class="confidence-label">{top3[0][1]*100:.1f}% de confiance</div>
            </div>
            """, unsafe_allow_html=True)

            # Séparateur
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Top 3 détaillé
            st.markdown('<div class="top3-header">Détail des prédictions</div>', unsafe_allow_html=True)
            for idx, (digit, conf) in enumerate(top3, 1):
                st.progress(float(conf), text=f"#{idx} - Chiffre {digit} : {conf*100:.1f}%")

            # Affichage du score de qualité
            display_quality_score(quality_score)

elif mode == "📸 Caméra":
    camera_input = st.camera_input("📸 Prendre une photo du chiffre")

    if camera_input:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-header">Photo capturée</div>', unsafe_allow_html=True)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image := Image.open(camera_input), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">Résultats de l\'analyse</div>', unsafe_allow_html=True)

            with st.spinner("🔍 Analyse en cours..."):
                top3, steps, quality_score = predict_mnist(
                    image, model,
                    return_steps=True,
                    rembg_model=rembg_model,
                    use_tta=use_tta,
                    return_quality=True
                )

            # Résultat principal
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prédiction</div>
                <div class="prediction-value">{top3[0][0]}</div>
                <div class="confidence-label">{top3[0][1]*100:.1f}% de confiance</div>
            </div>
            """, unsafe_allow_html=True)

            # Séparateur
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Top 3 détaillé
            st.markdown('<div class="top3-header">Détail des prédictions</div>', unsafe_allow_html=True)
            for idx, (digit, conf) in enumerate(top3, 1):
                st.progress(float(conf), text=f"#{idx} - Chiffre {digit} : {conf*100:.1f}%")

            # Affichage du score de qualité
            display_quality_score(quality_score)

elif mode == "✍️ Dessiner":
    st.markdown("**Dessinez un chiffre dans le canvas ci-dessous**")

    # Canvas de dessin
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Bouton pour lancer la prédiction
    predict_button = st.button("✨ Prédire le chiffre", type="primary", use_container_width=True)

    if canvas_result.image_data is not None and predict_button:
        # Vérifier si quelque chose a été dessiné
        if np.any(canvas_result.image_data[:, :, :3] != 255):
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown('<div class="section-header">Votre dessin</div>', unsafe_allow_html=True)
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(canvas_result.image_data, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="section-header">Résultats de l\'analyse</div>', unsafe_allow_html=True)

                with st.spinner("🔍 Analyse en cours..."):
                    # Convertir l'image du canvas en PIL
                    img_array = canvas_result.image_data[:, :, :3]
                    image = Image.fromarray(img_array.astype('uint8'), 'RGB')

                    top3, steps, quality_score = predict_mnist(
                        image, model,
                        return_steps=True,
                        rembg_model=rembg_model,
                        use_tta=use_tta,
                        return_quality=True
                    )

                # Résultat principal
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Prédiction</div>
                    <div class="prediction-value">{top3[0][0]}</div>
                    <div class="confidence-label">{top3[0][1]*100:.1f}% de confiance</div>
                </div>
                """, unsafe_allow_html=True)

                # Séparateur
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Top 3 détaillé
                st.markdown('<div class="top3-header">Détail des prédictions</div>', unsafe_allow_html=True)
                for idx, (digit, conf) in enumerate(top3, 1):
                    st.progress(float(conf), text=f"#{idx} - Chiffre {digit} : {conf*100:.1f}%")

                # Affichage du score de qualité
                display_quality_score(quality_score)
        else:
            st.info("👆 Dessinez un chiffre puis cliquez sur 'Prédire le chiffre'")
    elif canvas_result.image_data is not None:
        st.info("👆 Dessinez un chiffre puis cliquez sur 'Prédire le chiffre'")

# Footer
st.markdown("""
<div class="footer-note">
    <p style="margin-top: 0.5rem;">Développé par Pharnix ZONDO avec Statcrafter et Claude puis ChatGPT</p>
</div>
""", unsafe_allow_html=True)