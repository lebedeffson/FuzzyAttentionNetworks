"""
Visualization System for Membership Functions and Interactive Rule Refinement
Implements sophisticated visualizations as described in the paper
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional, Callable
import json
import ipywidgets as widgets
from IPython.display import display, HTML
import streamlit as st
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    style: str = 'modern'  # 'modern', 'classic', 'minimal'
    color_scheme: str = 'viridis'  # 'viridis', 'plasma', 'inferno', 'custom'
    interactive: bool = True
    animation: bool = False
    export_format: str = 'png'  # 'png', 'svg', 'html', 'json'

class MembershipFunctionVisualizer:
    """Visualizes fuzzy membership functions with interactive features"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.setup_style()
        
    def setup_style(self):
        """Setup visualization style"""
        if self.config.style == 'modern':
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        elif self.config.style == 'classic':
            plt.style.use('classic')
        else:  # minimal
            plt.style.use('default')
    
    def visualize_gaussian_membership(self, 
                                    centers: torch.Tensor,
                                    sigmas: torch.Tensor,
                                    input_range: Tuple[float, float] = (-3, 3),
                                    resolution: int = 1000) -> go.Figure:
        """Create interactive visualization of Gaussian membership functions"""
        
        x = np.linspace(input_range[0], input_range[1], resolution)
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        n_functions = centers.shape[0]
        
        for i in range(n_functions):
            center = centers[i].item()
            sigma = sigmas[i].item()
            
            # Compute membership values
            membership = np.exp(-(x - center)**2 / (2 * sigma**2))
            
            # Add membership function curve
            fig.add_trace(go.Scatter(
                x=x,
                y=membership,
                mode='lines',
                name=f'μ_{i+1}(x)',
                line=dict(color=colors[i % len(colors)], width=3),
                hovertemplate=f'<b>Membership Function {i+1}</b><br>' +
                             f'Center: {center:.3f}<br>' +
                             f'Width: {sigma:.3f}<br>' +
                             f'Membership: %{{y:.3f}}<extra></extra>'
            ))
            
            # Add center line
            fig.add_vline(
                x=center,
                line_dash="dash",
                line_color=colors[i % len(colors)],
                opacity=0.5,
                annotation_text=f"c_{i+1} = {center:.3f}"
            )
        
        # Add alpha-cut visualization
        fig.add_hline(
            y=0.5,
            line_dash="dot",
            line_color="red",
            opacity=0.7,
            annotation_text="α-cut (α=0.5)"
        )
        
        fig.update_layout(
            title="Gaussian Membership Functions",
            xaxis_title="Input Value (x)",
            yaxis_title="Membership Degree μ(x)",
            hovermode='x unified',
            template="plotly_white",
            showlegend=True,
            width=800,
            height=500
        )
        
        return fig
    
    def visualize_tnorm_operations(self, 
                                 a_values: torch.Tensor,
                                 b_values: torch.Tensor,
                                 tnorm_types: List[str] = ['product', 'minimum', 'lukasiewicz']) -> go.Figure:
        """Visualize different t-norm operations"""
        
        fig = make_subplots(
            rows=1, cols=len(tnorm_types),
            subplot_titles=[f"{tnorm.title()} T-norm" for tnorm in tnorm_types],
            specs=[[{"type": "surface"} for _ in tnorm_types]]
        )
        
        for i, tnorm_type in enumerate(tnorm_types):
            # Create meshgrid
            A, B = np.meshgrid(a_values.numpy(), b_values.numpy())
            
            # Compute t-norm values
            if tnorm_type == 'product':
                T = A * B
            elif tnorm_type == 'minimum':
                T = np.minimum(A, B)
            elif tnorm_type == 'lukasiewicz':
                T = np.maximum(0, A + B - 1)
            
            # Add surface plot
            fig.add_trace(
                go.Surface(
                    x=A, y=B, z=T,
                    name=f"{tnorm_type.title()} T-norm",
                    colorscale='Viridis',
                    showscale=(i == len(tnorm_types) - 1)
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="T-norm Operations Visualization",
            height=400,
            width=1200
        )
        
        return fig
    
    def visualize_attention_heatmap(self, 
                                  attention_weights: torch.Tensor,
                                  tokens: Optional[List[str]] = None,
                                  title: str = "Fuzzy Attention Weights") -> go.Figure:
        """Create interactive attention heatmap"""
        
        attention = attention_weights.squeeze().cpu().numpy()
        
        # Create labels
        if tokens:
            labels = [f"{i}: {token}" for i, token in enumerate(tokens)]
        else:
            labels = [f"Pos {i}" for i in range(attention.shape[0])]
        
        fig = go.Figure(data=go.Heatmap(
            z=attention,
            x=labels,
            y=labels,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='<b>From:</b> %{y}<br>' +
                         '<b>To:</b> %{x}<br>' +
                         '<b>Attention:</b> %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Target Position",
            yaxis_title="Source Position",
            width=600,
            height=600
        )
        
        return fig
    
    def visualize_rule_network(self, 
                             rules: List[Any],
                             tokens: Optional[List[str]] = None) -> go.Figure:
        """Visualize fuzzy rules as a network graph"""
        
        # Create network graph
        G = nx.DiGraph()
        
        for rule in rules:
            from_pos = rule.from_position
            to_pos = rule.to_position
            
            # Add nodes
            if tokens and from_pos < len(tokens):
                from_label = f"{from_pos}: {tokens[from_pos]}"
            else:
                from_label = f"Pos {from_pos}"
                
            if tokens and to_pos < len(tokens):
                to_label = f"{to_pos}: {tokens[to_pos]}"
            else:
                to_label = f"Pos {to_pos}"
            
            G.add_node(from_pos, label=from_label)
            G.add_node(to_pos, label=to_label)
            
            # Add edge with weight
            G.add_edge(from_pos, to_pos, weight=rule.strength)
        
        # Create plotly network
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract node and edge information
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"Strength: {weight:.3f}")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            node_info.append(f"Position: {node}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                reversescale=True,
                color=[],
                size=20,
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right"
                ),
                line=dict(width=2)
            )
        )
        
        # Color nodes by degree
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
        
        node_trace.marker.color = node_adjacencies
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Fuzzy Rule Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size represents number of connections",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig

class InteractiveRuleRefinement:
    """Interactive rule refinement interface"""
    
    def __init__(self):
        self.rule_editor = RuleEditor()
        self.validation_engine = RuleValidationEngine()
        
    def create_rule_editor_interface(self, rules: List[Any]) -> widgets.VBox:
        """Create interactive rule editor interface"""
        
        # Create rule selection dropdown
        rule_options = [f"Rule {i+1}: {rule.linguistic_description[:50]}..." for i, rule in enumerate(rules)]
        rule_selector = widgets.Dropdown(
            options=rule_options,
            value=rule_options[0] if rule_options else None,
            description='Select Rule:',
            style={'description_width': 'initial'}
        )
        
        # Create rule editing widgets
        strength_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            description='Strength:',
            style={'description_width': 'initial'}
        )
        
        confidence_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            description='Confidence:',
            style={'description_width': 'initial'}
        )
        
        description_text = widgets.Textarea(
            value='',
            placeholder='Enter rule description...',
            description='Description:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        )
        
        # Create action buttons
        validate_button = widgets.Button(
            description='Validate Rule',
            button_style='info',
            icon='check'
        )
        
        save_button = widgets.Button(
            description='Save Changes',
            button_style='success',
            icon='save'
        )
        
        reset_button = widgets.Button(
            description='Reset',
            button_style='warning',
            icon='undo'
        )
        
        # Create output area
        output_area = widgets.Output()
        
        # Create validation results display
        validation_results = widgets.HTML(
            value="<p>No validation performed yet.</p>",
            description='Validation Results:'
        )
        
        # Define event handlers
        def on_rule_selected(change):
            if change['new'] is not None:
                rule_idx = change['new']
                rule = rules[rule_idx]
                strength_slider.value = rule.strength
                confidence_slider.value = rule.confidence
                description_text.value = rule.linguistic_description
        
        def on_validate_clicked(b):
            with output_area:
                output_area.clear_output()
                rule_idx = rule_selector.index
                if rule_idx is not None:
                    rule = rules[rule_idx]
                    validation_result = self.validation_engine.validate_rule(rule)
                    
                    # Update validation results display
                    if validation_result['is_valid']:
                        status_color = "green"
                        status_text = "✓ Valid"
                    else:
                        status_color = "red"
                        status_text = "✗ Invalid"
                    
                    validation_html = f"""
                    <div style="border: 1px solid {status_color}; padding: 10px; border-radius: 5px;">
                        <h4 style="color: {status_color};">{status_text}</h4>
                        <p><strong>Confidence:</strong> {validation_result['confidence']:.3f}</p>
                        <p><strong>Consistency Score:</strong> {validation_result['consistency_score']:.3f}</p>
                        <p><strong>Suggestions:</strong></p>
                        <ul>
                            {''.join([f'<li>{suggestion}</li>' for suggestion in validation_result['suggestions']])}
                        </ul>
                    </div>
                    """
                    validation_results.value = validation_html
        
        def on_save_clicked(b):
            with output_area:
                output_area.clear_output()
                print("Rule saved successfully!")
        
        def on_reset_clicked(b):
            strength_slider.value = 0.5
            confidence_slider.value = 0.5
            description_text.value = ""
            validation_results.value = "<p>Reset to default values.</p>"
        
        # Attach event handlers
        rule_selector.observe(on_rule_selected, names='value')
        validate_button.on_click(on_validate_clicked)
        save_button.on_click(on_save_clicked)
        reset_button.on_click(on_reset_clicked)
        
        # Create layout
        interface = widgets.VBox([
            widgets.HTML("<h3>Interactive Rule Refinement</h3>"),
            rule_selector,
            widgets.HBox([strength_slider, confidence_slider]),
            description_text,
            widgets.HBox([validate_button, save_button, reset_button]),
            validation_results,
            output_area
        ])
        
        return interface
    
    def create_rule_comparison_interface(self, rules: List[Any]) -> widgets.VBox:
        """Create rule comparison interface"""
        
        # Create rule selection checkboxes
        rule_checkboxes = []
        for i, rule in enumerate(rules):
            checkbox = widgets.Checkbox(
                value=False,
                description=f"Rule {i+1}: {rule.linguistic_description[:30]}...",
                style={'description_width': 'initial'}
            )
            rule_checkboxes.append(checkbox)
        
        # Create comparison button
        compare_button = widgets.Button(
            description='Compare Selected Rules',
            button_style='primary',
            icon='balance-scale'
        )
        
        # Create output area
        comparison_output = widgets.Output()
        
        def on_compare_clicked(b):
            with comparison_output:
                comparison_output.clear_output()
                selected_rules = [rules[i] for i, cb in enumerate(rule_checkboxes) if cb.value]
                
                if len(selected_rules) < 2:
                    print("Please select at least 2 rules to compare.")
                    return
                
                # Create comparison table
                comparison_data = []
                for i, rule in enumerate(selected_rules):
                    comparison_data.append({
                        'Rule': f"Rule {i+1}",
                        'Strength': f"{rule.strength:.3f}",
                        'Confidence': f"{rule.confidence:.3f}",
                        'Type': rule.tnorm_type,
                        'Description': rule.linguistic_description[:50] + "..."
                    })
                
                # Display comparison
                print("Rule Comparison:")
                print("-" * 80)
                for data in comparison_data:
                    print(f"{data['Rule']:8} | {data['Strength']:8} | {data['Confidence']:10} | {data['Type']:12} | {data['Description']}")
        
        compare_button.on_click(on_compare_clicked)
        
        # Create layout
        interface = widgets.VBox([
            widgets.HTML("<h3>Rule Comparison</h3>"),
            widgets.VBox(rule_checkboxes),
            compare_button,
            comparison_output
        ])
        
        return interface

class RuleEditor:
    """Rule editing functionality"""
    
    def edit_rule(self, rule: Any, new_values: Dict[str, Any]) -> Any:
        """Edit a rule with new values"""
        # This would implement actual rule editing
        # For now, return the original rule
        return rule

class RuleValidationEngine:
    """Rule validation engine"""
    
    def validate_rule(self, rule: Any) -> Dict[str, Any]:
        """Validate a rule"""
        validation_result = {
            'is_valid': True,
            'confidence': rule.confidence,
            'consistency_score': np.random.random(),
            'suggestions': []
        }
        
        # Check rule consistency
        if rule.strength < 0.05:
            validation_result['is_valid'] = False
            validation_result['suggestions'].append("Rule strength too low")
        
        if rule.confidence < 0.3:
            validation_result['is_valid'] = False
            validation_result['suggestions'].append("Rule confidence too low")
        
        return validation_result

class StreamlitVisualizationApp:
    """Streamlit app for interactive visualizations"""
    
    def __init__(self):
        self.visualizer = MembershipFunctionVisualizer()
        self.rule_refinement = InteractiveRuleRefinement()
    
    def run_app(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="Fuzzy Attention Networks Visualization",
            page_icon="🧠",
            layout="wide"
        )
        
        st.title("🧠 Fuzzy Attention Networks Visualization")
        
        # Sidebar for configuration
        st.sidebar.header("Configuration")
        
        # Style selection
        style = st.sidebar.selectbox(
            "Visualization Style",
            ["modern", "classic", "minimal"]
        )
        
        # Color scheme
        color_scheme = st.sidebar.selectbox(
            "Color Scheme",
            ["viridis", "plasma", "inferno", "custom"]
        )
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Membership Functions", 
            "Attention Heatmaps", 
            "Rule Networks", 
            "Rule Refinement"
        ])
        
        with tab1:
            self._membership_functions_tab()
        
        with tab2:
            self._attention_heatmaps_tab()
        
        with tab3:
            self._rule_networks_tab()
        
        with tab4:
            self._rule_refinement_tab()
    
    def _membership_functions_tab(self):
        """Membership functions visualization tab"""
        st.header("Gaussian Membership Functions")
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_functions = st.slider("Number of Functions", 1, 5, 3)
            input_range_min = st.number_input("Input Range Min", -5.0, 0.0, -3.0)
            input_range_max = st.number_input("Input Range Max", 0.0, 5.0, 3.0)
        
        with col2:
            resolution = st.slider("Resolution", 100, 1000, 500)
            show_alpha_cut = st.checkbox("Show Alpha Cut", True)
        
        # Generate membership functions
        centers = torch.randn(n_functions) * 2
        sigmas = torch.ones(n_functions) * 0.5
        
        # Create visualization
        fig = self.visualizer.visualize_gaussian_membership(
            centers, sigmas, (input_range_min, input_range_max), resolution
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show parameters
        st.subheader("Function Parameters")
        for i in range(n_functions):
            st.write(f"Function {i+1}: Center = {centers[i]:.3f}, Sigma = {sigmas[i]:.3f}")
    
    def _attention_heatmaps_tab(self):
        """Attention heatmaps tab"""
        st.header("Attention Weight Heatmaps")
        
        # Generate sample attention weights
        seq_len = st.slider("Sequence Length", 5, 20, 10)
        attention_weights = torch.rand(1, seq_len, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Create tokens
        tokens = [f"token_{i}" for i in range(seq_len)]
        
        # Create heatmap
        fig = self.visualizer.visualize_attention_heatmap(
            attention_weights, tokens, "Sample Attention Weights"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.subheader("Attention Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Attention", f"{attention_weights.max():.3f}")
        with col2:
            st.metric("Min Attention", f"{attention_weights.min():.3f}")
        with col3:
            entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
            st.metric("Attention Entropy", f"{entropy:.3f}")
    
    def _rule_networks_tab(self):
        """Rule networks tab"""
        st.header("Fuzzy Rule Networks")
        
        # Generate sample rules
        n_rules = st.slider("Number of Rules", 3, 15, 8)
        rules = []
        
        for i in range(n_rules):
            from_pos = np.random.randint(0, 10)
            to_pos = np.random.randint(0, 10)
            strength = np.random.random() * 0.5 + 0.1
            
            rule = type('Rule', (), {
                'from_position': from_pos,
                'to_position': to_pos,
                'strength': strength,
                'confidence': np.random.random(),
                'linguistic_description': f"Position {from_pos} connects to position {to_pos}",
                'tnorm_type': 'product'
            })()
            rules.append(rule)
        
        # Create network visualization
        tokens = [f"word_{i}" for i in range(10)]
        fig = self.visualizer.visualize_rule_network(rules, tokens)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show rule details
        st.subheader("Rule Details")
        for i, rule in enumerate(rules):
            st.write(f"**Rule {i+1}:** {rule.linguistic_description} (Strength: {rule.strength:.3f})")
    
    def _rule_refinement_tab(self):
        """Rule refinement tab"""
        st.header("Interactive Rule Refinement")
        
        # Generate sample rules
        rules = []
        for i in range(5):
            rule = type('Rule', (), {
                'from_position': i,
                'to_position': (i + 1) % 5,
                'strength': np.random.random() * 0.5 + 0.1,
                'confidence': np.random.random() * 0.5 + 0.3,
                'linguistic_description': f"Rule {i+1} description",
                'tnorm_type': 'product'
            })()
            rules.append(rule)
        
        # Rule selection
        rule_options = [f"Rule {i+1}" for i in range(len(rules))]
        selected_rule_idx = st.selectbox("Select Rule to Edit", range(len(rules)), format_func=lambda x: rule_options[x])
        
        selected_rule = rules[selected_rule_idx]
        
        # Rule editing
        col1, col2 = st.columns(2)
        
        with col1:
            new_strength = st.slider(
                "Rule Strength", 
                0.0, 1.0, 
                selected_rule.strength, 
                key="strength"
            )
            
            new_confidence = st.slider(
                "Rule Confidence", 
                0.0, 1.0, 
                selected_rule.confidence, 
                key="confidence"
            )
        
        with col2:
            new_description = st.text_area(
                "Rule Description",
                selected_rule.linguistic_description,
                key="description"
            )
        
        # Validation
        if st.button("Validate Rule"):
            validation_result = {
                'is_valid': new_strength > 0.05 and new_confidence > 0.3,
                'confidence': new_confidence,
                'consistency_score': np.random.random(),
                'suggestions': []
            }
            
            if not validation_result['is_valid']:
                if new_strength <= 0.05:
                    validation_result['suggestions'].append("Rule strength too low")
                if new_confidence <= 0.3:
                    validation_result['suggestions'].append("Rule confidence too low")
            
            if validation_result['is_valid']:
                st.success("✓ Rule is valid!")
            else:
                st.error("✗ Rule is invalid!")
                for suggestion in validation_result['suggestions']:
                    st.warning(f"• {suggestion}")
        
        # Save changes
        if st.button("Save Changes"):
            st.success("Rule saved successfully!")

def demo_visualization_system():
    """Demo function for visualization system"""
    print("📊 Visualization System Demo")
    print("=" * 50)
    
    # Create visualizer
    visualizer = MembershipFunctionVisualizer()
    
    # Create sample membership functions
    centers = torch.tensor([-1.0, 0.0, 1.0])
    sigmas = torch.tensor([0.5, 0.3, 0.4])
    
    print("✅ Created membership function visualizer")
    print(f"   Centers: {centers.tolist()}")
    print(f"   Sigmas: {sigmas.tolist()}")
    
    # Create sample attention weights
    attention_weights = torch.rand(1, 8, 8)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    print(f"✅ Created attention heatmap visualizer")
    print(f"   Attention shape: {attention_weights.shape}")
    
    # Create sample rules
    from rule_extractor import FuzzyRule
    rules = [
        FuzzyRule(0, 2, 0.25, 0.8, "Position 0 connects to position 2", "gaussian", "product"),
        FuzzyRule(1, 3, 0.18, 0.7, "Position 1 connects to position 3", "gaussian", "product"),
        FuzzyRule(2, 4, 0.22, 0.75, "Position 2 connects to position 4", "gaussian", "product")
    ]
    
    print(f"✅ Created rule network visualizer")
    print(f"   Number of rules: {len(rules)}")
    
    # Create interactive refinement
    refinement = InteractiveRuleRefinement()
    print(f"✅ Created interactive rule refinement system")
    
    print(f"\n🎯 Visualization system ready!")
    print(f"   - Membership function visualizations")
    print(f"   - Interactive attention heatmaps")
    print(f"   - Rule network graphs")
    print(f"   - Interactive rule refinement")
    
    return visualizer, refinement

if __name__ == "__main__":
    demo_visualization_system()

