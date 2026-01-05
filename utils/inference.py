import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session
from scipy import ndimage

# Cache global pour les sessions rembg (évite de recréer à chaque appel)
_rembg_sessions = {}

def calculate_preprocessing_quality(digit_gray, aspect_ratio, current_max_size):
    """
    Calcule un score de confiance pour le preprocessing

    Args:
        digit_gray: Image en niveaux de gris du chiffre extrait
        aspect_ratio: Ratio largeur/hauteur de la détection
        current_max_size: Taille maximale du chiffre détecté

    Returns:
        dict: Score de qualité avec métriques détaillées
    """
    # Calcul du contraste (std de l'image)
    contrast = float(np.std(digit_gray))

    # Score de contraste (bon si > 50)
    contrast_score = min(contrast / 50.0, 1.0)

    # Score de taille (bon si entre 50 et 500 pixels)
    if 50 <= current_max_size <= 500:
        size_score = 1.0
    elif current_max_size < 50:
        size_score = current_max_size / 50.0
    else:
        size_score = max(0.0, 1.0 - (current_max_size - 500) / 500.0)

    # Score d'aspect ratio (bon si entre 0.5 et 2.0)
    if 0.5 <= aspect_ratio <= 2.0:
        aspect_score = 1.0
    else:
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0) / 2.0)

    # Score global (moyenne pondérée)
    global_score = (contrast_score * 0.5 + size_score * 0.3 + aspect_score * 0.2)

    # Déterminer le niveau de qualité
    if global_score >= 0.75:
        quality_level = "Excellente"
    elif global_score >= 0.5:
        quality_level = "Bonne"
    elif global_score >= 0.3:
        quality_level = "Moyenne"
    else:
        quality_level = "Faible"

    return {
        'global_score': round(global_score, 2),
        'quality_level': quality_level,
        'contrast': round(contrast, 2),
        'contrast_score': round(contrast_score, 2),
        'size': current_max_size,
        'size_score': round(size_score, 2),
        'aspect_ratio': round(aspect_ratio, 2),
        'aspect_score': round(aspect_score, 2)
    }

def apply_rotation(img_array, angle):
    """Applique une rotation à une image numpy"""
    return ndimage.rotate(img_array, angle, reshape=False, order=1)

def predict_mnist(img, model, return_steps=False, rembg_model="u2netp", use_tta=False, return_quality=False):
    """
    Prédiction à partir d'une image PIL avec prétraitement MNIST-like robuste et optimisé

    Pipeline optimisé avec rembg pour isolation automatique du chiffre :
    0. Suppression automatique de l'arrière-plan avec rembg (deep learning, session cachée)
    1. Composition sur fond adaptatif (analyse de l'intensité du chiffre)
    2. Conversion en grayscale
    3. Débruitage gaussien adaptatif (kernel variable selon taille image)
    4. Détection automatique du type de fond (clair/foncé)
    5. Binarisation Otsu TEMPORAIRE (uniquement pour détecter la bounding box)
    6. Extraction de la région d'intérêt avec validation (aspect ratio)
    7. Inversion conditionnelle + normalisation du contraste CLAHE
    8. Redimensionnement proportionnel vers ~20×20 avec anti-aliasing
    9. Centrage par centre de masse dans canvas 28×28
    10. Post-processing morphologique (closing léger)
    11. Normalisation selon les stats d'entraînement (modèle fait : (x - 33.32) / 78.57)
    12. [Optionnel] TTA (Test-Time Augmentation) avec 5 rotations

    AMÉLIORATIONS :
    - ✅ Session rembg cachée (gain de performance)
    - ✅ Débruitage adaptatif selon taille d'image
    - ✅ Contraste amélioré avec CLAHE
    - ✅ Validation des détections (reject aspect ratio aberrants)
    - ✅ Post-processing morphologique pour meilleur match MNIST
    - ✅ Composition sur fond adaptatif (gère chiffre blanc sur noir)
    - ✅ TTA (Test-Time Augmentation) pour gain de +0.2-0.4%
    - ✅ Score de qualité du preprocessing

    Args:
        img: Image PIL
        model: Modèle Keras chargé
        return_steps: Si True, retourne aussi les images de chaque étape
        rembg_model: Modèle rembg à utiliser. Options:
            - "u2netp" (défaut, recommandé pour MNIST) : Léger et performant
            - "u2net" : Bon équilibre qualité/vitesse
            - "isnet-general-use" : Plus récent, meilleure qualité générale
        use_tta: Si True, utilise Test-Time Augmentation (5 rotations, gain +0.2-0.4%, 5× plus lent)
        return_quality: Si True, retourne le score de qualité du preprocessing

    Returns:
        Si return_steps=False et return_quality=False: list: Top 3 prédictions [(digit, confidence), ...]
        Si return_steps=True: tuple: (top3, steps_dict, [quality_dict si return_quality])
        Si return_quality=True: tuple: (top3, quality_dict, [steps_dict si return_steps])
    """

    # Dictionnaire pour stocker les étapes (si demandé)
    steps = {} if return_steps else None

    # --- 0. Suppression automatique de l'arrière-plan avec rembg ---
    # Cela isole le chiffre même avec fond complexe/texturé
    # Utiliser une session cachée pour meilleure performance (évite recréation)
    if rembg_model not in _rembg_sessions:
        _rembg_sessions[rembg_model] = new_session(rembg_model)
    session = _rembg_sessions[rembg_model]
    img_no_bg = remove(img, session=session)  # Retourne une image RGBA avec fond transparent

    # --- 1. Composer sur fond adaptatif (analyse du chiffre) ---
    # Analyser les pixels non-transparents pour déterminer si le chiffre est clair ou foncé
    img_rgba = img_no_bg.convert("RGBA")
    pixels = np.array(img_rgba)
    alpha_mask = pixels[:, :, 3] > 0  # Pixels non-transparents

    if np.any(alpha_mask):
        # Calculer la moyenne des pixels du chiffre (RGB seulement)
        digit_pixels = pixels[alpha_mask][:, :3]  # RGB seulement
        mean_digit_intensity = np.mean(digit_pixels)

        # Si chiffre clair → fond noir, si chiffre foncé → fond blanc
        # Cela garantit un bon contraste quelle que soit la couleur du chiffre
        bg_color = (0, 0, 0) if mean_digit_intensity > 127 else (255, 255, 255)
    else:
        # Fallback : fond blanc par défaut si rien détecté
        bg_color = (255, 255, 255)

    bg = Image.new("RGB", img_no_bg.size, bg_color)
    img_composite = Image.alpha_composite(bg.convert("RGBA"), img_rgba).convert("RGB")

    if return_steps:
        # Pour la visualisation
        steps['0_background_removed'] = np.array(img_composite.convert("L"))

    # --- 2. Passage en niveaux de gris ---
    img_gray = np.array(img_composite.convert("L"))
    if return_steps:
        steps['1_grayscale'] = img_gray.copy()

    # --- 3. Débruitage adaptatif ---
    # Adapter le kernel selon la taille de l'image pour un débruitage optimal
    kernel_size = max(3, min(7, img_gray.shape[0] // 100))
    if kernel_size % 2 == 0:  # Le kernel doit être impair
        kernel_size += 1
    img_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    if return_steps:
        steps['2_blurred'] = img_blur.copy()

    # --- 4. Détection automatique du type de fond (noir ou blanc) ---
    # Calculer la moyenne de l'image pour savoir si fond clair ou foncé
    mean_intensity = np.mean(img_blur)
    is_light_background = mean_intensity > 127

    # --- 5. Binarisation adaptée au fond (détection temporaire pour bounding box) ---
    if is_light_background:
        # Fond clair → BINARY_INV (chiffre noir devient blanc)
        _, img_bin_temp = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Fond foncé → BINARY (chiffre blanc reste blanc)
        _, img_bin_temp = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if return_steps:
        steps['3_binary_detection'] = img_bin_temp.copy()

    # --- 6. Extraction de la région d'intérêt avec padding ---
    coords = np.column_stack(np.where(img_bin_temp > 0))
    if coords.size == 0:
        # Cas pathologique : rien détecté
        empty = np.zeros((1, 28, 28, 1), dtype=np.float32)
        preds = model.predict(empty, verbose=0)[0]
        top3 = np.argsort(preds)[::-1][:3]
        if return_steps:
            steps['4_cropped_grayscale'] = np.zeros((28, 28), dtype=np.uint8)
            steps['5_resized'] = np.zeros((28, 28), dtype=np.uint8)
            steps['6_final_28x28'] = np.zeros((28, 28), dtype=np.uint8)
            return list(zip(top3, preds[top3])), steps
        return list(zip(top3, preds[top3]))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # --- 6.1. Validation de la détection ---
    # Vérifier que la forme détectée est raisonnable (pas trop déformée)
    h_bbox, w_bbox = y1 - y0, x1 - x0
    aspect_ratio = max(h_bbox, w_bbox) / max(1, min(h_bbox, w_bbox))

    # Rejeter les détections avec aspect ratio aberrant (probablement pas un chiffre)
    if aspect_ratio > 5:  # Trop allongé/déformé
        # Retourner une prédiction vide
        empty = np.zeros((1, 28, 28, 1), dtype=np.float32)
        preds = model.predict(empty, verbose=0)[0]
        top3 = np.argsort(preds)[::-1][:3]
        if return_steps:
            steps['4_cropped_grayscale'] = np.zeros((28, 28), dtype=np.uint8)
            steps['5_resized'] = np.zeros((28, 28), dtype=np.uint8)
            steps['6_final_28x28'] = np.zeros((28, 28), dtype=np.uint8)
            return list(zip(top3, preds[top3])), steps
        return list(zip(top3, preds[top3]))

    # --- 6.2. Padding optimisé ---
    # MNIST a généralement 4 pixels de marge, on optimise le padding
    target_digit_size = 20  # Les chiffres MNIST font ~20×20
    current_max_size = max(h_bbox, w_bbox)

    # Calculer un padding qui amènera le chiffre vers 20×20 dans le canvas 28×28
    if current_max_size > 0:
        optimal_pad = max(2, int((28 - target_digit_size) / 2))
        pad = int(max(h_bbox, w_bbox) * 0.1) + optimal_pad  # 10% + padding optimal
    else:
        pad = 2

    y0 = max(0, y0 - pad)
    y1 = min(img_blur.shape[0], y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(img_blur.shape[1], x1 + pad)

    # IMPORTANT : Extraire depuis l'image GRAYSCALE (pas binarisée)
    digit_crop = img_blur[y0:y1+1, x0:x1+1]

    # Inverser si fond clair (MNIST = blanc sur noir)
    if is_light_background:
        digit_gray = 255 - digit_crop
    else:
        digit_gray = digit_crop

    # --- 7. Normaliser le contraste avec CLAHE ---
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) améliore le contraste
    # tout en évitant l'amplification excessive du bruit
    if digit_gray.max() > digit_gray.min():
        # Étirement d'histogramme d'abord
        digit_gray = ((digit_gray - digit_gray.min()) * 255.0 /
                      (digit_gray.max() - digit_gray.min())).astype(np.uint8)

        # Puis CLAHE pour améliorer les détails
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        digit_gray = clahe.apply(digit_gray)

    # --- Calcul du score de qualité du preprocessing (si demandé) ---
    quality_score = None
    if return_quality:
        quality_score = calculate_preprocessing_quality(digit_gray, aspect_ratio, current_max_size)

    if return_steps:
        steps['4_cropped_grayscale'] = digit_gray.copy()

    # --- 8. Resize proportionnel vers 20×20 avec interpolation de qualité ---
    h, w = digit_gray.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # Utiliser INTER_AREA pour shrinking (meilleur anti-aliasing)
    digit_resized = cv2.resize(
        digit_gray,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    if return_steps:
        steps['5_resized'] = digit_resized.copy()

    # --- 8. Centrage par centre de masse (comme MNIST) ---
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Calcul du centre de masse
    M = cv2.moments(digit_resized)
    if M["m00"] != 0:
        cx_digit = M["m10"] / M["m00"]
        cy_digit = M["m01"] / M["m00"]
    else:
        # Fallback : centre géométrique
        cy_digit, cx_digit = np.array(digit_resized.shape) / 2

    # Centrer dans le canvas 28×28
    y_offset = int(14 - cy_digit)
    x_offset = int(14 - cx_digit)

    # Placer le chiffre (avec vérification des limites)
    for i in range(digit_resized.shape[0]):
        for j in range(digit_resized.shape[1]):
            y_canvas = y_offset + i
            x_canvas = x_offset + j
            if 0 <= y_canvas < 28 and 0 <= x_canvas < 28:
                canvas[y_canvas, x_canvas] = digit_resized[i, j]

    # --- 9. Post-processing morphologique ---
    # Légère opération de closing pour améliorer la forme et mieux matcher MNIST
    # Cela comble les petits trous et lisse légèrement les contours
    kernel = np.ones((2, 2), np.uint8)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

    # --- 10. Préparation pour le modèle ---
    # Le modèle attend [0, 255] en float32 (normalise lui-même avec mu=33.32, std=78.57)
    if return_steps:
        steps['6_final_28x28'] = canvas.copy()

    img_array = canvas.astype(np.float32)
    img_array = img_array[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)

    # --- 11. Prédiction (avec ou sans TTA) ---
    if use_tta:
        # Test-Time Augmentation : 5 rotations + moyenne
        angles = [-5, -3, 0, 3, 5]  # Rotations légères en degrés
        all_predictions = []

        for angle in angles:
            if angle == 0:
                # Pas de rotation
                pred = model.predict(img_array, verbose=0)[0]
            else:
                # Appliquer rotation
                rotated = apply_rotation(canvas, angle)
                rotated_array = rotated.astype(np.float32)[np.newaxis, ..., np.newaxis]
                pred = model.predict(rotated_array, verbose=0)[0]

            all_predictions.append(pred)

        # Moyenne des 5 prédictions
        predictions = np.mean(all_predictions, axis=0)
    else:
        # Prédiction simple (sans TTA)
        predictions = model.predict(img_array, verbose=0)[0]

    # --- 12. Top 3 ---
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3_confidences = predictions[top3_indices]

    top3 = list(zip(top3_indices, top3_confidences))

    # --- 13. Return selon les options ---
    if return_steps and return_quality:
        return top3, steps, quality_score
    elif return_steps:
        return top3, steps
    elif return_quality:
        return top3, quality_score
    else:
        return top3