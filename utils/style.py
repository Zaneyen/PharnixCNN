"""
Module de style centralisé pour l'application Streamlit.
"""
import streamlit as st

def apply_style():
    st.markdown("""
<style>
    /* --- Configuration Générale --- */
    html, body, [class*="css"], .stApp {
        color: #1f2937;
    }

    /* Fond blanc pur */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Animation d'entrée */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-enter {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    .delay-100 { animation-delay: 0.1s; }
    .delay-200 { animation-delay: 0.2s; }
    .delay-300 { animation-delay: 0.3s; }

    /* --- Cards & Glassmorphism --- */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        
        /* Layout Flex pour hauteur uniforme dans les colonnes */
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        border-color: rgba(37, 99, 235, 0.3);
        background: rgba(255, 255, 255, 0.85);
    }
    
    /* Titres dans les cartes */
    .glass-card h3 {
        color: #1e3a8a;
        font-weight: 600;
        margin-top: 0;
    }
    
    /* --- Fix Layout : Hauteur égale des cartes --- */
    [data-testid="column"] {
        display: flex;
        flex-direction: column; 
    }
    
    [data-testid="column"] > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="column"] .glass-card {
        flex: 1; 
        height: auto; 
    }

    /* --- Métriques --- */
    .metric-container {
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* --- Grille Responsive pour les Chiffres MNIST --- */
    .mnist-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
        gap: 10px;
        margin-top: 1rem;
        width: 100%;
    }
    
    .mnist-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        background: rgba(255,255,255,0.4);
        padding: 5px;
        border-radius: 8px;
    }
    
    .mnist-item img {
        width: 100%;
        max-width: 100%;
        border-radius: 4px;
        display: block;
    }
    
    .mnist-label {
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 5px;
    }

    /* --- Avatar Auteur --- */
    .author-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        display: block;
        margin: 0 auto;
    }

</style>
""", unsafe_allow_html=True)

# --- Fonctions Helper ---

def create_card(title, content, icon=None):
    """Génère le HTML pour une carte stylisée sans interférence Markdown."""
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    return f'<div class="glass-card animate-enter"><h3 style="margin-top: 0; margin-bottom: 1rem; font-size: 1.25rem; display: flex; align-items: center;">{icon_html}{title.strip()}</h3><div style="color: #4b5563; line-height: 1.6;">{content.strip()}</div></div>'

def create_metric(value, label, delta=None):
    """Génère le HTML pour une métrique sans interférence Markdown."""
    delta_html = f'<div style="color: #10b981; font-size: 0.8rem; margin-top: 0.25rem;">{delta}</div>' if delta else ""
    return f'<div class="glass-card metric-container animate-enter" style="justify-content: center;"><div class="metric-value">{value}</div><div class="metric-label">{label}</div>{delta_html}</div>'

def create_link_card(title, content, icon, target_url):
    """Crée une carte cliquable avec un rendu propre."""
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    return f'<a href="{target_url}" target="_self" style="text-decoration: none; color: inherit; display: block;"><div class="glass-card animate-enter" style="cursor: pointer; height: 100%;"><h3 style="margin-top: 0; margin-bottom: 1rem; font-size: 1.25rem; display: flex; align-items: center; color: #1e3a8a;">{icon_html}{title.strip()}</h3><div style="color: #4b5563; line-height: 1.6;">{content.strip()}</div></div></a>'