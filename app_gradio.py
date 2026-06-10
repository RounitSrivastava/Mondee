#new persona
"""
app_gradio.py

Gradio Checkpoint Interface for Persona-Based Recommendation Engine

Features:
- Interactive persona selection / custom definition
- Real-time recommendation generation
- Score visualization
- Batch recommendation export

Run:
    python app_gradio.py
    
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import json
import os
from datetime import datetime
from typing import List, Tuple

from core.recommender import recommend

# ============================================================================
# PREDEFINED PERSONAS (checkpoint templates)
# ============================================================================

CHECKPOINT_PERSONAS = {
    "🚗 Truck Driver": {
        "user_id": "user_trucker_001",
        "persona": "truck_driver",
        "signals": ["automotive", "gps", "dash_cam", "long_distance"],
        "query_string": "truck driver electronics gps navigation",
        "destination": ""
    },
    
    "💰 Budget Shopper": {
        "user_id": "user_budget_001",
        "persona": "budget_buyer",
        "signals": ["affordable", "discount", "value"],
        "query_string": "cheap electronics deals",
        "destination": ""
    },
    
    "👑 Premium Buyer": {
        "user_id": "user_premium_001",
        "persona": "premium_buyer",
        "signals": ["luxury", "high_quality", "premium"],
        "query_string": "premium electronics luxury",
        "destination": ""
    },
    
    "🏃 Sports Enthusiast": {
        "user_id": "user_sports_001",
        "persona": "sports_enthusiast",
        "signals": ["fitness", "athletic", "performance"],
        "query_string": "sports fitness electronics",
        "destination": ""
    },
    
    "🥗 Health Conscious": {
        "user_id": "user_health_001",
        "persona": "health_conscious",
        "signals": ["wellness", "organic", "nutrition"],
        "query_string": "health wellness electronics",
        "destination": ""
    },
    
    "🎁 Gift Buyer": {
        "user_id": "user_gift_001",
        "persona": "gift_buyer",
        "signals": ["gift", "birthday", "present"],
        "query_string": "gift ideas electronics",
        "destination": ""
    },
    
    "👨‍👩‍👧‍👦 Family Buyer": {
        "user_id": "user_family_001",
        "persona": "family_buyer",
        "signals": ["family", "kids", "household"],
        "query_string": "family household electronics",
        "destination": ""
    },
    
    "🔧 Tech Geek": {
        "user_id": "user_techgeek_001",
        "persona": "tech_geek",
        "signals": ["latest", "innovation", "cutting_edge"],
        "query_string": "latest technology gadgets",
        "destination": ""
    },
    
    "📚 Student": {
        "user_id": "user_student_001",
        "persona": "student",
        "signals": ["laptop", "productivity", "learning"],
        "query_string": "student laptop study electronics",
        "destination": ""
    }
}

# ============================================================================
# RECOMMENDATION FUNCTION
# ============================================================================

def generate_recommendations(
    persona_preset: str,
    custom_persona_name: str = "",
    custom_signals: str = "",
    custom_query: str = "",
    top_k: int = 15
) -> Tuple[str, str]:
    """
    Generate recommendations for a selected or custom persona.
    
    Returns:
        (recommendations_table, status_message)
    """
    
    try:
        # Determine which persona to use
        if persona_preset == "Custom":
            if not custom_persona_name or not custom_query:
                return (
                    "<div style='color:red;'><b>Error:</b> Please provide persona name and query.</div>",
                    "❌ Custom persona incomplete"
                )
            
            persona = {
                "user_id": "user_custom_001",
                "persona": custom_persona_name.lower().replace(" ", "_"),
                "signals": [s.strip() for s in custom_signals.split(",") if s.strip()],
                "query_string": custom_query,
                "destination": ""
            }
        else:
            persona = CHECKPOINT_PERSONAS.get(
                persona_preset,
                CHECKPOINT_PERSONAS["💰 Budget Shopper"]
            )
        
        # Generate recommendations
        result = recommend(persona, top_k=top_k)
        
        # Format as HTML table
        recommendations = result.get("recommendations", [])
        
        if not recommendations:
            return (
                "<div style='color:orange;'><b>No recommendations found.</b></div>",
                "⚠️ Empty result set"
            )
        
        html_table = "<table style='width:100%; border-collapse: collapse;'>"
        html_table += "<tr style='background-color: #f0f0f0;'>"
        html_table += "<th style='border: 1px solid #ddd; padding: 10px; text-align:left;'>Rank</th>"
        html_table += "<th style='border: 1px solid #ddd; padding: 10px; text-align:left;'>Product Name</th>"
        html_table += "<th style='border: 1px solid #ddd; padding: 10px; text-align:left;'>Details</th>"
        html_table += "<th style='border: 1px solid #ddd; padding: 10px; text-align:center;'>Score</th>"
        html_table += "</tr>"
        
        for rec in recommendations:
            rank = rec.get("rank", "?")
            name = rec.get("name", "N/A")
            details = rec.get("details", "")[:100] + "..." if rec.get("details") else ""
            score = rec.get("score", 0)
            
            # Score color coding
            if score >= 80:
                score_color = "#28a745"  # green
            elif score >= 60:
                score_color = "#ffc107"  # yellow
            else:
                score_color = "#dc3545"  # red
            
            html_table += f"<tr>"
            html_table += f"<td style='border: 1px solid #ddd; padding: 8px;'><b>#{rank}</b></td>"
            html_table += f"<td style='border: 1px solid #ddd; padding: 8px;'>{name}</td>"
            html_table += f"<td style='border: 1px solid #ddd; padding: 8px;'>{details}</td>"
            html_table += f"<td style='border: 1px solid #ddd; padding: 8px; text-align:center; color:{score_color}; font-weight:bold;'>{score:.1f}%</td>"
            html_table += f"</tr>"
        
        html_table += "</table>"
        
        status = (
            f"✅ Generated {len(recommendations)} recommendations\n"
            f"Persona: {result.get('persona', 'unknown')}\n"
            f"Query: {result.get('query_string_used', '')}\n"
            f"Generated: {result.get('generated_at', 'unknown')}"
        )
        
        return (html_table, status)
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return (
            f"<div style='color:red;'><b>{error_msg}</b></div>",
            error_msg
        )

# ============================================================================
# EXPORT FUNCTION
# ============================================================================

def export_recommendations(
    persona_preset: str,
    custom_persona_name: str = "",
    custom_signals: str = "",
    custom_query: str = "",
    top_k: int = 15
) -> str:
    """
    Export recommendations as JSON.
    """
    
    try:
        # Determine which persona to use
        if persona_preset == "Custom":
            if not custom_persona_name or not custom_query:
                return "Error: Please complete the custom persona first."
            
            persona = {
                "user_id": "user_custom_001",
                "persona": custom_persona_name.lower().replace(" ", "_"),
                "signals": [s.strip() for s in custom_signals.split(",") if s.strip()],
                "query_string": custom_query,
                "destination": ""
            }
        else:
            persona = CHECKPOINT_PERSONAS.get(persona_preset)
        
        # Generate recommendations
        result = recommend(persona, top_k=top_k)
        
        # Export as JSON
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{export_dir}/recommendations_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return f"✅ Exported to {filename}"
    
    except Exception as e:
        return f"❌ Export failed: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(
        title="Mondee Recommendation Engine Checkpoint",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown(
            """
            # 🎯 Mondee Recommendation Engine
            ### Persona-Based Product Discovery
            
            This checkpoint demonstrates the hybrid retrieval recommendation system using:
            - **E5-Large-v2** embeddings (768-dim semantic vectors)
            - **LDA** topic modeling (30-dim topic distribution)
            - **FAISS IVF** index (optimized nearest-neighbor search)
            
            Select a predefined persona or create a custom one to explore personalized recommendations.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Persona Configuration")
                
                persona_dropdown = gr.Dropdown(
                    choices=list(CHECKPOINT_PERSONAS.keys()) + ["Custom"],
                    value="💰 Budget Shopper",
                    label="Preset Personas",
                    info="Choose a template or 'Custom' to define your own"
                )
                
                with gr.Group(visible=False) as custom_group:
                    custom_name = gr.Textbox(
                        label="Persona Name",
                        placeholder="e.g., Adventure Seeker",
                        info="Name for your custom persona"
                    )
                    custom_signals = gr.Textbox(
                        label="Signals (comma-separated)",
                        placeholder="e.g., outdoor, adventure, hiking",
                        info="Behavioral signals / keywords"
                    )
                    custom_query = gr.Textbox(
                        label="Query String",
                        placeholder="e.g., outdoor adventure gear",
                        info="Main search query"
                    )
                
                def toggle_custom(choice):
                    return gr.update(visible=(choice == "Custom"))
                
                persona_dropdown.change(
                    toggle_custom,
                    inputs=persona_dropdown,
                    outputs=custom_group
                )
                
                top_k_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    step=5,
                    value=15,
                    label="Top-K Results",
                    info="Number of recommendations to generate"
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate Recommendations",
                    variant="primary",
                    scale=1
                )
                
                export_btn = gr.Button(
                    "📥 Export as JSON",
                    variant="secondary",
                    scale=1
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                rec_table = gr.HTML(
                    value="<p style='color: #999;'>Recommendations will appear here...</p>"
                )
                
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
                
                export_output = gr.Textbox(
                    label="Export Status",
                    interactive=False,
                    visible=False
                )
        
        # Event handlers
        generate_btn.click(
            fn=generate_recommendations,
            inputs=[
                persona_dropdown,
                custom_name,
                custom_signals,
                custom_query,
                top_k_slider
            ],
            outputs=[rec_table, status_box]
        )
        
        export_btn.click(
            fn=export_recommendations,
            inputs=[
                persona_dropdown,
                custom_name,
                custom_signals,
                custom_query,
                top_k_slider
            ],
            outputs=export_output
        )
        
        gr.Markdown(
            """
            ---
            
            ### 📚 About This Checkpoint
            
            **Model Architecture:**
            - Vectorizer: RoBERTa + LDA (798-dim combined)
            - Indexer: FAISS IVF with Inner Product metric
            - Batch size: 128 | LDA components: 30
            
            **Data Pipeline:**
            1. `scripts/build_index.py` — ingests Amazon Electronics metadata
            2. `core/indexer.py` — builds FAISS index with hybrid vectors
            3. `core/recommender.py` — retrieves & ranks recommendations
            4. This app — provides interactive checkpoint interface
            
            **How to Use:**
            1. Select a persona or define a custom one
            2. Adjust top-K slider for more/fewer results
            3. Click "Generate Recommendations"
            4. Export results as JSON for further analysis
            """
        )
    
    return interface

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        
        share=True,
        
    )
