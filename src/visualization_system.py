"""
Visualization System for Fuzzy Attention Networks
Interactive visualizations for attention patterns and fuzzy rules
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass

from rule_extractor import FuzzyRule

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    figure_size: Tuple[int, int] = (12, 8)
    color_scheme: str = 'viridis'
    font_size: int = 12
    dpi: int = 300
    interactive: bool = True

class AttentionVisualizer:
    """Visualize attention patterns and fuzzy rules"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.setup_style()
        
    def setup_style(self):
        """Setup matplotlib and seaborn styles"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.config.color_scheme)
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'figure.dpi': self.config.dpi,
            'savefig.dpi': self.config.dpi
        })
    
    def plot_attention_heatmap(self, 
                              attention_weights: torch.Tensor,
                              tokens: Optional[List[str]] = None,
                              title: str = "Attention Heatmap",
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot attention weights as heatmap"""
        
        # Convert to numpy and average over batch dimension
        if attention_weights.dim() == 3:
            attention = attention_weights.mean(dim=0).cpu().numpy()
        else:
            attention = attention_weights.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create heatmap
        im = ax.imshow(attention, cmap='Blues', aspect='auto')
        
        # Set labels
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        else:
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Set title
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.dpi)
        
        return fig
    
    def plot_fuzzy_rules_network(self, 
                                rules: List[FuzzyRule],
                                tokens: Optional[List[str]] = None,
                                title: str = "Fuzzy Rules Network",
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot fuzzy rules as network graph"""
        
        if not rules:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'No fuzzy rules to display', 
                   ha='center', va='center', fontsize=16)
            ax.set_title(title)
            return fig
        
        # Create network data
        nodes = set()
        edges = []
        edge_weights = []
        
        for rule in rules:
            nodes.add(rule.from_position)
            nodes.add(rule.to_position)
            edges.append((rule.from_position, rule.to_position))
            edge_weights.append(rule.strength)
        
        nodes = sorted(list(nodes))
        node_labels = {}
        if tokens:
            for i, node in enumerate(nodes):
                if node < len(tokens):
                    node_labels[node] = tokens[node]
                else:
                    node_labels[node] = f"Pos {node}"
        else:
            node_labels = {node: f"Pos {node}" for node in nodes}
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Position nodes in a circle
        n_nodes = len(nodes)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Plot edges
        for i, (from_pos, to_pos) in enumerate(edges):
            from_idx = nodes.index(from_pos)
            to_idx = nodes.index(to_pos)
            
            x_coords = [x_pos[from_idx], x_pos[to_idx]]
            y_coords = [y_pos[from_idx], y_pos[to_idx]]
            
            # Line width based on rule strength
            line_width = max(1, edge_weights[i] * 10)
            
            ax.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=line_width)
        
        # Plot nodes
        ax.scatter(x_pos, y_pos, s=200, c='lightblue', edgecolors='black', linewidth=2)
        
        # Add labels
        for i, node in enumerate(nodes):
            ax.annotate(node_labels[node], (x_pos[i], y_pos[i]), 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Set title and remove axes
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend for edge weights
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Strong Rule'),
            plt.Line2D([0], [0], color='blue', linewidth=1, label='Weak Rule')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.dpi)
        
        return fig
    
    def plot_rule_strength_distribution(self, 
                                       rules: List[FuzzyRule],
                                       title: str = "Rule Strength Distribution",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of rule strengths"""
        
        if not rules:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'No fuzzy rules to display', 
                   ha='center', va='center', fontsize=16)
            ax.set_title(title)
            return fig
        
        strengths = [rule.strength for rule in rules]
        confidences = [rule.confidence for rule in rules]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # Strength histogram
        ax1.hist(strengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Rule Strength')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Rule Strength Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax2.hist(confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Rule Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Rule Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.dpi)
        
        return fig
    
    def plot_attention_entropy(self, 
                              attention_weights: torch.Tensor,
                              title: str = "Attention Entropy",
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot attention entropy for each position"""
        
        # Calculate entropy for each position
        if attention_weights.dim() == 3:
            attention = attention_weights.mean(dim=0)  # Average over batch
        else:
            attention = attention_weights
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=-1)
        entropy = entropy.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        positions = range(len(entropy))
        bars = ax.bar(positions, entropy, color='lightblue', edgecolor='black')
        
        # Color bars based on entropy level
        for i, bar in enumerate(bars):
            if entropy[i] > np.percentile(entropy, 75):
                bar.set_color('red')
            elif entropy[i] > np.percentile(entropy, 50):
                bar.set_color('orange')
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Attention Entropy')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_entropy = np.mean(entropy)
        ax.axhline(y=mean_entropy, color='red', linestyle='--', 
                  label=f'Mean: {mean_entropy:.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.dpi)
        
        return fig

class InteractiveVisualizer:
    """Interactive visualizations using Plotly"""
    
    def __init__(self):
        self.color_schemes = {
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'inferno': px.colors.sequential.Inferno,
            'magma': px.colors.sequential.Magma
        }
    
    def create_interactive_attention_heatmap(self, 
                                  attention_weights: torch.Tensor,
                                  tokens: Optional[List[str]] = None,
                                           title: str = "Interactive Attention Heatmap") -> go.Figure:
        """Create interactive attention heatmap with Plotly"""
        
        # Convert to numpy and average over batch dimension
        if attention_weights.dim() == 3:
            attention = attention_weights.mean(dim=0).cpu().numpy()
        else:
            attention = attention_weights.cpu().numpy()
        
        # Create hover text
        if tokens:
            hover_text = []
            for i in range(len(tokens)):
                row = []
                for j in range(len(tokens)):
                    row.append(f"Query: {tokens[i]}<br>Key: {tokens[j]}<br>Attention: {attention[i, j]:.4f}")
                hover_text.append(row)
        else:
            hover_text = None
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention,
            x=tokens if tokens else None,
            y=tokens if tokens else None,
            hoverongaps=False,
            hovertext=hover_text,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Attention Weight")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Key Position",
            yaxis_title="Query Position",
            width=800,
            height=600
        )
        
        return fig
    
    def create_interactive_rule_network(self, 
                                      rules: List[FuzzyRule],
                                      tokens: Optional[List[str]] = None,
                                      title: str = "Interactive Fuzzy Rules Network") -> go.Figure:
        """Create interactive network visualization of fuzzy rules"""
        
        if not rules:
            fig = go.Figure()
            fig.add_annotation(
                text="No fuzzy rules to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=title)
            return fig
        
        # Create network data
        nodes = set()
        edges = []
        
        for rule in rules:
            nodes.add(rule.from_position)
            nodes.add(rule.to_position)
            edges.append({
                'from': rule.from_position,
                'to': rule.to_position,
                'strength': rule.strength,
                'confidence': rule.confidence,
                'description': rule.linguistic_description
            })
        
        nodes = sorted(list(nodes))
        
        # Position nodes in a circle
        n_nodes = len(nodes)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Create node data
        node_data = []
        for i, node in enumerate(nodes):
            label = tokens[node] if tokens and node < len(tokens) else f"Pos {node}"
            node_data.append({
                'x': x_pos[i],
                'y': y_pos[i],
                'label': label,
                'position': node
            })
        
        # Create edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in edges:
            from_idx = nodes.index(edge['from'])
            to_idx = nodes.index(edge['to'])
            
            edge_x.extend([x_pos[from_idx], x_pos[to_idx], None])
            edge_y.extend([y_pos[from_idx], y_pos[to_idx], None])
            
            edge_info.append({
                'from': edge['from'],
                'to': edge['to'],
                'strength': edge['strength'],
                'confidence': edge['confidence'],
                'description': edge['description']
            })
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='rgba(0,0,255,0.5)'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=[node['x'] for node in node_data],
            y=[node['y'] for node in node_data],
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='black')),
            text=[node['label'] for node in node_data],
            textposition="middle center",
            hoverinfo='text',
            hovertext=[f"Position: {node['position']}<br>Label: {node['label']}" for node in node_data],
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                text="Hover over nodes and edges for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        return fig

    def create_rule_comparison_chart(self, 
                                   rules_by_level: Dict[str, List[FuzzyRule]],
                                   title: str = "Rule Comparison by User Level") -> go.Figure:
        """Create comparison chart of rules across different user levels"""
        
        # Prepare data
        data = []
        for level, rules in rules_by_level.items():
            for rule in rules:
                data.append({
                    'User Level': level,
                    'Rule Strength': rule.strength,
                    'Rule Confidence': rule.confidence,
                    'Rule ID': f"{rule.from_position}-{rule.to_position}"
                })
        
        if not data:
            fig = go.Figure()
            fig.add_annotation(
                text="No rules to compare",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=title)
            return fig
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Rule Strength by User Level', 'Rule Confidence by User Level'),
            specs=[[{"type": "box"}, {"type": "box"}]]
        )
        
        # Add box plots
        for level in df['User Level'].unique():
            level_data = df[df['User Level'] == level]
            
            fig.add_trace(
                go.Box(
                    y=level_data['Rule Strength'],
                    name=f"{level} Strength",
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Box(
                    y=level_data['Rule Confidence'],
                    name=f"{level} Confidence",
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            width=1000,
            height=500
        )
        
        return fig

def demo_visualization_system():
    """Demo function for visualization system"""
    print("🎨 Visualization System Demo")
    print("=" * 50)
    
    # Create sample data
    seq_len = 8
    attention_weights = torch.rand(1, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly", "watching"]
    
    # Create sample rules
    rules = [
        FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
        FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
        FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product"),
        FuzzyRule(2, 5, 0.15, 0.65, "Position 2 moderately attends to position 5", "gaussian", "product"),
        FuzzyRule(3, 7, 0.10, 0.55, "Position 3 slightly attends to position 7", "gaussian", "product")
    ]
    
    # Test matplotlib visualizer
    print("📊 Testing Matplotlib Visualizer...")
    visualizer = AttentionVisualizer()
    
    # Attention heatmap
    fig1 = visualizer.plot_attention_heatmap(attention_weights, tokens, "Sample Attention Heatmap")
    print("✅ Attention heatmap created")
    
    # Fuzzy rules network
    fig2 = visualizer.plot_fuzzy_rules_network(rules, tokens, "Sample Fuzzy Rules Network")
    print("✅ Fuzzy rules network created")
    
    # Rule strength distribution
    fig3 = visualizer.plot_rule_strength_distribution(rules, "Sample Rule Strength Distribution")
    print("✅ Rule strength distribution created")
    
    # Attention entropy
    fig4 = visualizer.plot_attention_entropy(attention_weights, "Sample Attention Entropy")
    print("✅ Attention entropy plot created")
    
    # Test interactive visualizer
    print("\n🎯 Testing Interactive Visualizer...")
    interactive_viz = InteractiveVisualizer()
    
    # Interactive attention heatmap
    fig5 = interactive_viz.create_interactive_attention_heatmap(attention_weights, tokens, "Interactive Attention Heatmap")
    print("✅ Interactive attention heatmap created")
    
    # Interactive rule network
    fig6 = interactive_viz.create_interactive_rule_network(rules, tokens, "Interactive Fuzzy Rules Network")
    print("✅ Interactive rule network created")
    
    # Rule comparison chart
    rules_by_level = {
        'novice': rules[:2],
        'intermediate': rules[:4],
        'expert': rules
    }
    fig7 = interactive_viz.create_rule_comparison_chart(rules_by_level, "Rule Comparison by User Level")
    print("✅ Rule comparison chart created")
    
    print("\n🎉 All visualizations created successfully!")
    print("💡 Use the returned figures to display or save visualizations")
    
    return visualizer, interactive_viz

if __name__ == "__main__":
    demo_visualization_system()