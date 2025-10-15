#!/usr/bin/env python3
"""
Отладка графиков fuzzy membership функций
"""

import numpy as np
import plotly.graph_objects as go
from demos.final_working_interface import load_fuzzy_membership_functions
import warnings
warnings.filterwarnings('ignore')

def debug_fuzzy_plot(dataset_name):
    """Отлаживаем график fuzzy функций для конкретного датасета"""
    
    print(f"=== Отладка {dataset_name} ===")
    
    # Загружаем fuzzy параметры
    fuzzy_params = load_fuzzy_membership_functions(dataset_name)
    
    print(f"Центры: {fuzzy_params['centers']}")
    print(f"Ширины: {fuzzy_params['widths']}")
    
    # Определяем правильный диапазон x на основе центров
    centers = fuzzy_params['centers']
    widths = fuzzy_params['widths']
    
    min_center = min(centers)
    max_center = max(centers)
    max_width = max(widths)
    
    # Расширяем диапазон для полного отображения функций
    x_min = min_center - 3 * max_width
    x_max = max_center + 3 * max_width
    
    print(f"Диапазон центров: {min_center:.3f} до {max_center:.3f}")
    print(f"Максимальная ширина: {max_width:.3f}")
    print(f"Рекомендуемый диапазон x: {x_min:.3f} до {x_max:.3f}")
    
    # Создаем x с правильным диапазоном
    x = np.linspace(x_min, x_max, 1000)
    
    # Названия нечетких множеств
    fuzzy_set_names = [
        "Text: Semantic Similarity",
        "Text: Word Importance", 
        "Text: Context Relevance",
        "Image: Visual Saliency",
        "Image: Object Boundaries",
        "Image: Color Patterns",
        "Attention: Cross-modal Alignment"
    ]
    
    fig = go.Figure()
    
    for i, (center, width) in enumerate(zip(centers, widths)):
        # Bell membership function: μ(x) = 1 / (1 + ((x - c) / σ)²)
        y = 1 / (1 + ((x - center) / width) ** 2)
        set_name = fuzzy_set_names[i] if i < len(fuzzy_set_names) else f"Fuzzy Set {i + 1}"
        
        print(f"Функция {i+1}: центр={center:.3f}, ширина={width:.3f}, max_y={max(y):.3f}")
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=set_name,
            line=dict(width=3)
        ))
    
    title = f"Fuzzy Membership Functions - {dataset_name} (from {fuzzy_params['source']})"
    fig.update_layout(
        title=title,
        xaxis_title="Feature Value (x)",
        yaxis_title="Membership Degree μ(x)",
        height=500,
        xaxis=dict(
            title="Feature Value (x)",
            showgrid=True,
            gridcolor='lightgray',
            range=[x_min, x_max]
        ),
        yaxis=dict(
            title="Membership Degree μ(x)",
            range=[0, 1.1],
            showgrid=True,
            gridcolor='lightgray'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Сохраняем график
    filename = f"debug_fuzzy_{dataset_name}.html"
    fig.write_html(filename)
    print(f"График сохранен: {filename}")
    
    return fig

if __name__ == "__main__":
    # Тестируем все датасеты
    datasets = ['stanford_dogs', 'cifar10', 'ham10000']
    
    for dataset in datasets:
        debug_fuzzy_plot(dataset)
        print()

