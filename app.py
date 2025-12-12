import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ================================
# CONFIGURATION DE LA PAGE
# ================================
st.set_page_config(
    page_title="CrowdFunding recommendation - Recommandation de projets",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# STYLES CSS
# ================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        cursor: pointer;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .project-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .project-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.12);
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
    }
    
    .login-card {
        max-width: 500px;
        margin: 3rem auto;
        padding: 2.5rem;
        border-radius: 20px;
        background: white;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border: 1px solid #e8e8e8;
    }
    
    .login-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .login-subtitle {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-success {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-danger {
        background: #f8d7da;
        color: #721c24;
    }
    
    .badge-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-info {
        background: #d1ecf1;
        color: #0c5460;
    }
    
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .sidebar-profile {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 10px;
        margin: 0.8rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.5s ease;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# CHARGEMENT DES DONNÉES
# ================================
@st.cache_resource
def load_model_and_data():
    try:
        knn_model = joblib.load("knn_model.pkl")
        ratings_matrix_sparse = joblib.load("ratings_matrix_sparse.pkl")
        user_to_idx = joblib.load("user_to_idx.pkl")
        unique_users = joblib.load("unique_users.pkl")
        unique_projects = joblib.load("unique_projects.pkl")
        df = pd.read_csv("clean_projects_fixed.csv")
        inter = pd.read_csv("interactions.csv")

        # Conversion de types
        if isinstance(unique_users, (np.ndarray, list)):
            unique_users = [int(u) for u in unique_users]

        inter["user_id"] = inter["user_id"].astype(int)
        inter["project_id"] = inter["project_id"].astype(int)

        df["project_id"] = df["project_id"].astype(int)
        # On supprime rating et user_id dans df pour éviter les conflits au merge
        for col in ["rating", "user_id"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        return knn_model, ratings_matrix_sparse, user_to_idx, unique_users, unique_projects, df, inter

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None, None, None, None, None, None, None

# ================================
# GÉNÉRATION / LECTURE DE users.csv
# ================================
def get_or_create_users_df(inter, unique_users, users_csv_path="users.csv"):
    """
    Crée (ou lit) users.csv au format :
    user_id,password
    i,mdp_user_i
    """
    # IDs valides = intersection interactions / modèle
    ids_interactions = set(inter["user_id"].unique().tolist())
    ids_modele = set(int(u) for u in unique_users)
    ids_valides = sorted(list(ids_interactions & ids_modele))

    if len(ids_valides) == 0:
        raise ValueError("Aucun user_id valide trouvé (intersection vide entre interactions et unique_users).")

    # Si le fichier existe, on essaie de le lire
    if os.path.exists(users_csv_path):
        users_df = pd.read_csv(users_csv_path)
        if "user_id" not in users_df.columns or "password" not in users_df.columns:
            # mauvais format => on recrée proprement
            users_df = pd.DataFrame({"user_id": ids_valides})
            users_df["password"] = users_df["user_id"].apply(lambda uid: f"mdp_user_{uid}")
        else:
            users_df["user_id"] = users_df["user_id"].astype(int)
            # On filtre pour ne garder que les IDs encore valides
            users_df = users_df[users_df["user_id"].isin(ids_valides)]
            # Et on ajoute les éventuels nouveaux IDs (si interactions/unique_users ont évolué)
            missing_ids = sorted(list(set(ids_valides) - set(users_df["user_id"].tolist())))
            if missing_ids:
                to_add = pd.DataFrame({
                    "user_id": missing_ids,
                    "password": [f"mdp_user_{uid}" for uid in missing_ids]
                })
                users_df = pd.concat([users_df, to_add], ignore_index=True)
    else:
        # Création initiale
        users_df = pd.DataFrame({"user_id": ids_valides})
        users_df["password"] = users_df["user_id"].apply(lambda uid: f"mdp_user_{uid}")

    # On sauvegarde toujours au format CSV propre
    users_df.to_csv(users_csv_path, index=False)

    return users_df

# ================================
# MOTEUR DE RECOMMANDATION
# ================================
def recommander_user_based(user_id, knn_model, ratings_matrix_sparse, user_to_idx, unique_users, inter, n=5):
    if user_id not in user_to_idx:
        return []

    user_idx = user_to_idx[user_id]
    distances, indices = knn_model.kneighbors(
        ratings_matrix_sparse[user_idx],
        n_neighbors=min(n + 1, len(unique_users))
    )

    similar_indices = indices.flatten()[1:]
    recommended_projects = set()

    for idx in similar_indices:
        similar_user_id = unique_users[idx]
        projs = inter[inter["user_id"] == similar_user_id]["project_id"].tolist()
        recommended_projects.update(projs)
        if len(recommended_projects) >= n * 2:
            break

    return list(recommended_projects)[:n * 2]

# ================================
# COMPOSANTS UI
# ================================
def get_state_badge(state: str) -> str:
    state_lower = state.lower()
    if state_lower == "successful":
        cls = "badge-success"
        label = "RÉUSSI"
    elif state_lower == "failed":
        cls = "badge-danger"
        label = "ÉCHOUÉ"
    elif state_lower == "canceled":
        cls = "badge-warning"
        label = "ANNULÉ"
    elif state_lower == "suspended":
        cls = "badge-warning"
        label = "SUSPENDU"
    elif state_lower == "live":
        cls = "badge-info"
        label = "EN COURS"
    else:
        cls = "badge-info"
        label = state.upper()
    return f'<span class="badge {cls}">{label}</span>'


def display_project_card(project, index, show_rating=False, user_rating=None, show_state=False):
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {index}. {project.get('name', 'Projet inconnu')}")
        
        category = project.get('category', 'N/A')
        main_cat = project.get('main_category', 'N/A')
        country = project.get('country', 'N/A')
        st.markdown(f"**{main_cat}** • {category} • {country}")
        
        goal = float(project.get("goal", 0))
        pledged = float(project.get("pledged", 0))
        progress = min(pledged / goal, 1.0) if goal > 0 else 0
        
        st.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress * 100}%"></div>
            </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Objectif", f"{goal:,.0f} $")
        with col_b:
            st.metric("Collecté", f"{pledged:,.0f} $")
        with col_c:
            st.metric("Contributeurs", f"{int(project.get('backers', 0)):,}")
        with col_d:
            percent = (pledged / goal * 100) if goal > 0 else 0
            st.metric("Taux de financement", f"{percent:.0f} %")
    
    with col2:
        if show_state:
            state = str(project.get("state", "inconnu"))
            st.markdown(get_state_badge(state), unsafe_allow_html=True)
        
        if "duration_days" in project.index:
            days = int(project["duration_days"])
            st.markdown(f"Durée : **{days}** jours")
        
        if show_rating and user_rating is not None:
            st.markdown(f"Votre note : **{user_rating}/5**")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# CHARGEMENT
# ================================
knn_model, ratings_matrix_sparse, user_to_idx, unique_users, unique_projects, df, inter = load_model_and_data()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ================================
# ÉCRAN DE CONNEXION
# ================================
if df is not None and knn_model is not None and unique_users is not None and inter is not None:
    if not st.session_state.authenticated:
        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            st.markdown('<div class="login-title">CrowdFunding recommendation </div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="login-subtitle">Sélectionnez votre identifiant utilisateur et saisissez votre mot de passe pour accéder aux recommandations personnalisées</div>',
                unsafe_allow_html=True
            )

            try:
                users_df = get_or_create_users_df(inter, unique_users, users_csv_path="users.csv")
            except Exception as e:
                st.error(f"Erreur lors de la préparation de users.csv : {e}")
                st.stop()

            # dictionnaire user_id -> password
            users_db = dict(zip(users_df["user_id"], users_df["password"]))
            available_ids = sorted(users_db.keys())

            if len(available_ids) == 0:
                st.error("Aucun utilisateur disponible pour la connexion.")
                st.stop()

            selected_user_id = st.selectbox(
                "Identifiant utilisateur",
                options=available_ids,
                index=0,
                key="login_user_select"
            )

            mot_de_passe = st.text_input(
                "Mot de passe ",
                type="password",
                key="login_password"
            )

            if st.button("Se connecter", use_container_width=True):
                user_id_int = int(selected_user_id)
                expected_password = users_db.get(user_id_int)

                if mot_de_passe == expected_password:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_id_int
                    st.experimental_rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")

            st.markdown('</div>', unsafe_allow_html=True)

        st.stop()
else:
    st.error("Impossible de charger les données. Veuillez vérifier les fichiers nécessaires.")
    st.stop()

# ================================
# APPLICATION PRINCIPALE
# ================================
user_id = st.session_state.user_id

st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>
        CrowdFunding recommendation
    </h1>
    <p style='text-align: center; color: #6c757d; font-size: 1.1rem;'>
        Recommandations de projets
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ================================
# BARRE LATÉRALE
# ================================
with st.sidebar:
    user_stats = inter[inter["user_id"] == user_id]
    if "rating" in user_stats.columns:
        rating_series_sidebar = user_stats["rating"].dropna()
        avg_rating = rating_series_sidebar.mean() if len(rating_series_sidebar) > 0 else 0.0
    else:
        avg_rating = 0.0
    
    st.markdown(f"""
        <div class="sidebar-profile">
            <div style="font-size: 1rem; margin-bottom: 0.5rem;">Utilisateur connecté</div>
            <div style="font-size: 1.2rem; font-weight: 600;">Utilisateur #{user_id}</div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.3rem;">
                Note moyenne : {avg_rating:.1f}/5
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Se déconnecter", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.experimental_rerun()

    st.markdown("---")

    st.subheader("Paramètres de recommandation")
    n_recommandations = st.slider(
        "Nombre de recommandations",
        min_value=3,
        max_value=30,
        value=10,
        help="Nombre de projets à afficher"
    )

    st.markdown("---")

    st.subheader("Filtres")
    
    categories_list = sorted(df["main_category"].dropna().unique().tolist())
    countries_list = sorted(df["country"].dropna().unique().tolist())

    filtre_categorie = st.checkbox("Filtrer par catégorie")
    categorie_selectionnee = None
    if filtre_categorie:
        categorie_selectionnee = st.selectbox("Catégorie", ["Toutes"] + categories_list)
        if categorie_selectionnee == "Toutes":
            categorie_selectionnee = None

    filtre_pays = st.checkbox("Filtrer par pays")
    pays_selectionne = None
    if filtre_pays:
        pays_selectionne = st.selectbox("Pays", ["Tous"] + countries_list)
        if pays_selectionne == "Tous":
            pays_selectionne = None

    st.markdown("---")

    st.subheader("Votre activité")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Projets évalués", len(user_stats))
    with col_s2:
        if len(user_stats) > 0:
            distinct_cat = user_stats.merge(df, on="project_id", how="left")["main_category"].nunique()
        else:
            distinct_cat = 0
        st.metric("Catégories distinctes", distinct_cat)

# ================================
# ONGLET PRINCIPAUX
# ================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Recommandations",
    "Historique",
    "Tendances",
    "Analytique"
])

# ONGLET 1 : RECOMMANDATIONS
with tab1:
    st.subheader("Recommandations personnalisées")
    st.markdown("Générées à partir de vos évaluations et des utilisateurs similaires.")
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        generer_btn = st.button("Générer les recommandations", use_container_width=True)
    
    if generer_btn:
        with st.spinner("Génération des recommandations en cours..."):
            try:
                recommended_ids = recommander_user_based(
                    user_id, knn_model, ratings_matrix_sparse,
                    user_to_idx, unique_users, inter, n=n_recommandations
                )

                if recommended_ids:
                    recommended_projects = df[df["project_id"].isin(recommended_ids)].copy()

                    if categorie_selectionnee:
                        recommended_projects = recommended_projects[
                            recommended_projects["main_category"] == categorie_selectionnee
                        ]
                    if pays_selectionne:
                        recommended_projects = recommended_projects[
                            recommended_projects["country"] == pays_selectionne
                        ]

                    if len(recommended_projects) > 0:
                        recommended_projects = recommended_projects.sort_values(
                            "backers", ascending=False
                        )
                        
                        st.info(f"{len(recommended_projects)} projets trouvés avec les paramètres actuels.")
                        st.markdown("---")
                        
                        for idx, (_, project) in enumerate(recommended_projects.head(n_recommandations).iterrows(), 1):
                            display_project_card(project, idx, show_state=False)
                    else:
                        st.warning("Aucun projet ne correspond aux filtres sélectionnés. Essayez d’élargir les critères.")
                else:
                    st.info("Pas encore de recommandations disponibles. Il faut davantage d’évaluations.")

            except Exception as e:
                st.error(f"Erreur lors de la génération des recommandations : {e}")

# ONGLET 2 : HISTORIQUE
with tab2:
    st.subheader("Historique de vos évaluations")
    
    try:
        user_history = inter[inter["user_id"] == user_id].merge(
            df, on="project_id", how="left"
        )

        if len(user_history) == 0:
            st.info("Vous n’avez encore évalué aucun projet.")
        else:
            if "rating" in user_history.columns:
                rating_series = user_history["rating"].dropna()
            else:
                rating_series = pd.Series(dtype=float)

            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.metric("Nombre total d’évaluations", len(user_history))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                if len(rating_series) > 0:
                    moy = rating_series.mean()
                    st.metric("Note moyenne", f"{moy:.2f}/5")
                else:
                    st.metric("Note moyenne", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                categories = user_history["main_category"].nunique()
                st.metric("Catégories distinctes", categories)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                successful_mask = user_history["state"] == "successful"
                failed_mask = user_history["state"] == "failed"
                successful_count = int(successful_mask.sum())
                failed_count = int(failed_mask.sum())
                st.metric("Projets réussis / échoués", f"{successful_count} / {failed_count}")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            hist_left, hist_right = st.columns(2)

            with hist_left:
                st.markdown("Distribution des notes")
                if len(rating_series) > 0:
                    rating_counts = rating_series.value_counts().sort_index()
                    fig_hist = px.bar(
                        x=rating_counts.index.astype(str),
                        y=rating_counts.values,
                        labels={"x": "Note", "y": "Nombre"}
                    )
                    fig_hist.update_layout(height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.write("Aucune note disponible.")

            with hist_right:
                st.markdown("Statuts des projets évalués")
                state_counts = user_history["state"].value_counts()
                if len(state_counts) > 0:
                    fig_state = px.pie(
                        values=state_counts.values,
                        names=state_counts.index,
                        hole=0.4
                    )
                    fig_state.update_layout(height=300)
                    st.plotly_chart(fig_state, use_container_width=True)
                else:
                    st.write("Aucun statut disponible.")

            st.markdown("---")

            st.markdown("Historique détaillé")

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                tri_par = st.selectbox(
                    "Trier par",
                    ["Note (décroissante)", "Note (croissante)", "Plus récent (si disponible)"]
                )
            with col_f2:
                if len(rating_series) > 0:
                    min_note = float(rating_series.min())
                    max_note = float(rating_series.max())
                    if min_note > max_note:
                        min_note, max_note = max_note, min_note
                    filtre_note_min = st.slider(
                        "Note minimale",
                        min_value=min_note,
                        max_value=max_note,
                        value=min_note,
                        step=0.5
                    )
                else:
                    filtre_note_min = None
                    st.write("Impossible de filtrer par note (aucune note disponible).")
            with col_f3:
                max_affichage = st.slider("Nombre d’éléments à afficher", 5, 50, 20)

            if filtre_note_min is not None and len(rating_series) > 0:
                user_history_filtered = user_history[user_history["rating"] >= filtre_note_min]
            else:
                user_history_filtered = user_history.copy()

            if tri_par == "Note (décroissante)" and "rating" in user_history_filtered.columns:
                user_history_filtered = user_history_filtered.sort_values("rating", ascending=False)
            elif tri_par == "Note (croissante)" and "rating" in user_history_filtered.columns:
                user_history_filtered = user_history_filtered.sort_values("rating", ascending=True)
            else:
                if "timestamp" in user_history_filtered.columns:
                    user_history_filtered = user_history_filtered.sort_values("timestamp", ascending=False)
                else:
                    user_history_filtered = user_history_filtered.sort_values("project_id", ascending=False)

            for idx, (_, project) in enumerate(user_history_filtered.head(max_affichage).iterrows(), 1):
                display_project_card(
                    project,
                    idx,
                    show_rating=("rating" in project.index),
                    user_rating=project.get("rating", None),
                    show_state=True
                )

    except Exception as e:
        st.error(f"Erreur lors de l’affichage de l’historique : {e}")

# ONGLET 3 : TENDANCES
with tab3:
    st.subheader("Projets tendance")
    st.markdown("Les projets les plus populaires de la plateforme.")
    
    option_tri = st.radio(
        "Trier par",
        ["Plus de contributeurs", "Montant collecté le plus élevé", "Réussites récentes"],
        horizontal=True
    )
    
    if option_tri == "Plus de contributeurs":
        top_projects = df.nlargest(15, "backers")
    elif option_tri == "Montant collecté le plus élevé":
        top_projects = df.nlargest(15, "pledged")
    else:
        successful_global_df = df[df["state"] == "successful"].copy()
        if "launched" in successful_global_df.columns:
            successful_global_df["launched"] = pd.to_datetime(successful_global_df["launched"], errors='coerce')
            top_projects = successful_global_df.nlargest(15, "launched")
        else:
            top_projects = successful_global_df.nlargest(15, "backers")

    for idx, (_, project) in enumerate(top_projects.iterrows(), 1):
        display_project_card(project, idx, show_state=False)

# ONGLET 4 : ANALYTIQUE
with tab4:
    st.subheader("Analytique globale de la plateforme")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("Nombre total de projets", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        successful_global = len(df[df["state"] == "successful"])
        st.metric("Projets réussis", f"{successful_global:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        total_pledged_global = df["pledged"].sum()
        st.metric("Montant total collecté", f"{total_pledged_global/1e9:.2f} Md $")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        taux_reussite = (successful_global / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Taux de réussite global", f"{taux_reussite:.1f} %")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top catégories")
        category_data = df["main_category"].value_counts().head(10)
        fig = px.bar(
            x=category_data.values,
            y=category_data.index,
            orientation='h',
            labels={"x": "Nombre de projets", "y": "Catégorie"}
        )
        fig.update_layout(
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Projets par pays")
        country_data = df["country"].value_counts().head(10)
        fig = px.pie(
            values=country_data.values,
            names=country_data.index,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Répartition des statuts des projets")
    state_data = df["state"].value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=state_data.index,
            y=state_data.values
        )
    ])
    fig.update_layout(
        height=400,
        xaxis_title="Statut",
        yaxis_title="Nombre de projets"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #6c757d;'>
        <p style='font-size: 0.9rem;'>
            Propulsé par un modèle KNN • Développé avec Streamlit
        </p>
        <p style='font-size: 0.8rem; opacity: 0.8;'>
            © 2024 CrowdFund AI
        </p>
    </div>
""", unsafe_allow_html=True)
