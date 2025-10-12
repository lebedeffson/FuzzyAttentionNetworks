#!/usr/bin/env python3
"""
Генератор схемы архитектуры FAN для статьи уровня A
Создает профессиональную диаграмму через Graphviz
"""

from graphviz import Digraph
import os

def create_fan_architecture_diagram():
    """Создает схему архитектуры Fuzzy Attention Networks"""
    
    # Создаем граф с настройками для статьи
    dot = Digraph(comment='FAN Architecture', format='png')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='12', shape='box', style='filled')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Цветовая схема
    input_color = '#E3F2FD'      # Светло-синий для входов
    text_color = '#FFF3E0'       # Светло-оранжевый для текстовых компонентов
    image_color = '#E8F5E8'      # Светло-зеленый для изображений
    attention_color = '#F3E5F5'  # Светло-фиолетовый для attention
    fuzzy_color = '#FFEBEE'      # Светло-красный для fuzzy компонентов
    output_color = '#E0F2F1'     # Светло-бирюзовый для выходов
    fusion_color = '#FFF8E1'     # Светло-желтый для fusion
    
    # Входные данные
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(style='filled', color='lightgray', label='Input Layer')
        c.attr(fontsize='14', fontname='Arial Bold')
        
        dot.node('text_input', 'Text Input\n"Golden retriever dog"', 
                fillcolor=input_color, shape='ellipse')
        dot.node('image_input', 'Image Input\n224×224×3', 
                fillcolor=input_color, shape='ellipse')
    
    # Текстовый путь
    with dot.subgraph(name='cluster_text') as c:
        c.attr(style='filled', color='lightgray', label='Text Processing Pipeline')
        c.attr(fontsize='14', fontname='Arial Bold')
        
        dot.node('bert_tokenizer', 'BERT Tokenizer\n[CLS] golden retriever [SEP]', 
                fillcolor=text_color)
        dot.node('bert_encoder', 'BERT Encoder\n12 Layers, 768-dim', 
                fillcolor=text_color)
        dot.node('text_projection', 'Text Projection\nLinear(768→512)', 
                fillcolor=text_color)
    
    # Изображение путь
    with dot.subgraph(name='cluster_image') as c:
        c.attr(style='filled', color='lightgray', label='Image Processing Pipeline')
        c.attr(fontsize='14', fontname='Arial Bold')
        
        dot.node('resnet_backbone', 'ResNet Backbone\nResNet50/ResNet18', 
                fillcolor=image_color)
        dot.node('image_projection', 'Image Projection\nLinear(2048→512)', 
                fillcolor=image_color)
    
    # Fuzzy Attention механизм
    with dot.subgraph(name='cluster_fuzzy_attention') as c:
        c.attr(style='filled', color='lightgray', label='Fuzzy Attention Mechanism')
        c.attr(fontsize='14', fontname='Arial Bold')
        
        dot.node('query_proj', 'Query Projection\nLinear(512→512)', 
                fillcolor=attention_color)
        dot.node('key_proj', 'Key Projection\nLinear(512→512)', 
                fillcolor=attention_color)
        dot.node('value_proj', 'Value Projection\nLinear(512→512)', 
                fillcolor=attention_color)
        
        dot.node('fuzzy_centers', 'Fuzzy Centers\nμ₁, μ₂, ..., μₖ', 
                fillcolor=fuzzy_color, shape='diamond')
        dot.node('fuzzy_widths', 'Fuzzy Widths\nσ₁, σ₂, ..., σₖ', 
                fillcolor=fuzzy_color, shape='diamond')
        
        dot.node('bell_membership', 'Bell Membership\nμ(x) = 1/(1+((x-μ)/σ)²)', 
                fillcolor=fuzzy_color, shape='ellipse')
        
        dot.node('attention_weights', 'Attention Weights\nSoftmax(QKᵀ/√d)', 
                fillcolor=attention_color)
        dot.node('attended_values', 'Attended Values\nAttention × V', 
                fillcolor=attention_color)
    
    # Мультимодальное слияние
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(style='filled', color='lightgray', label='Multimodal Fusion')
        c.attr(fontsize='14', fontname='Arial Bold')
        
        dot.node('concat_fusion', 'Concatenation\n[text; image; attention]', 
                fillcolor=fusion_color)
        dot.node('fusion_layer', 'Fusion Layer\nLinear(1536→512)', 
                fillcolor=fusion_color)
        dot.node('dropout', 'Dropout\np=0.1', 
                fillcolor=fusion_color)
        dot.node('relu', 'ReLU Activation', 
                fillcolor=fusion_color)
    
    # Классификатор
    with dot.subgraph(name='cluster_classifier') as c:
        c.attr(style='filled', color='lightgray', label='Classification Head')
        c.attr(fontsize='14', fontname='Arial Bold')
        
        dot.node('classifier', 'Classifier\nLinear(512→num_classes)', 
                fillcolor=output_color)
        dot.node('softmax', 'Softmax\nProbability Distribution', 
                fillcolor=output_color)
        dot.node('prediction', 'Prediction\nClass + Confidence', 
                fillcolor=output_color, shape='ellipse')
    
    # Соединения - текстовый путь
    dot.edge('text_input', 'bert_tokenizer', label='Tokenize')
    dot.edge('bert_tokenizer', 'bert_encoder', label='Encode')
    dot.edge('bert_encoder', 'text_projection', label='Project')
    
    # Соединения - изображение путь
    dot.edge('image_input', 'resnet_backbone', label='Extract Features')
    dot.edge('resnet_backbone', 'image_projection', label='Project')
    
    # Соединения - attention механизм
    dot.edge('text_projection', 'query_proj', label='Q')
    dot.edge('text_projection', 'key_proj', label='K')
    dot.edge('text_projection', 'value_proj', label='V')
    
    dot.edge('fuzzy_centers', 'bell_membership', label='μ')
    dot.edge('fuzzy_widths', 'bell_membership', label='σ')
    dot.edge('query_proj', 'attention_weights', label='Q')
    dot.edge('key_proj', 'attention_weights', label='K')
    dot.edge('bell_membership', 'attention_weights', label='Fuzzy Modulate')
    dot.edge('attention_weights', 'attended_values', label='Attention')
    dot.edge('value_proj', 'attended_values', label='V')
    
    # Соединения - fusion
    dot.edge('text_projection', 'concat_fusion', label='Text Features')
    dot.edge('image_projection', 'concat_fusion', label='Image Features')
    dot.edge('attended_values', 'concat_fusion', label='Attended Features')
    
    dot.edge('concat_fusion', 'fusion_layer', label='Fuse')
    dot.edge('fusion_layer', 'dropout', label='Regularize')
    dot.edge('dropout', 'relu', label='Activate')
    
    # Соединения - классификация
    dot.edge('relu', 'classifier', label='Classify')
    dot.edge('classifier', 'softmax', label='Normalize')
    dot.edge('softmax', 'prediction', label='Output')
    
    # Добавляем математические формулы
    dot.node('formula1', 'Attention(Q,K,V) = softmax(QK^T/√d + F(μ,σ))V', 
            fillcolor='white', shape='plaintext', fontsize='10')
    dot.node('formula2', 'F(μ,σ) = Σᵢ 1/(1+((x-μᵢ)/σᵢ)²)', 
            fillcolor='white', shape='plaintext', fontsize='10')
    
    # Соединяем формулы
    dot.edge('formula1', 'attention_weights', style='dashed', color='gray')
    dot.edge('formula2', 'bell_membership', style='dashed', color='gray')
    
    return dot

def create_attention_head_diagram():
    """Создает детальную схему attention head"""
    
    dot = Digraph(comment='Attention Head Detail', format='png')
    dot.attr(rankdir='LR', size='10,8', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='11', shape='box', style='filled')
    
    # Цвета
    input_color = '#E3F2FD'
    process_color = '#F3E5F5'
    fuzzy_color = '#FFEBEE'
    output_color = '#E0F2F1'
    
    # Входы
    dot.node('Q', 'Query\nQ ∈ R^(d×d)', fillcolor=input_color)
    dot.node('K', 'Key\nK ∈ R^(d×d)', fillcolor=input_color)
    dot.node('V', 'Value\nV ∈ R^(d×d)', fillcolor=input_color)
    
    # Attention computation
    dot.node('QK', 'QK^T\nAttention Scores', fillcolor=process_color)
    dot.node('scale', 'Scale\n÷√d', fillcolor=process_color)
    
    # Fuzzy modulation
    dot.node('fuzzy_mod', 'Fuzzy Modulation\nF(μ,σ)', fillcolor=fuzzy_color)
    dot.node('softmax', 'Softmax\nNormalization', fillcolor=process_color)
    
    # Output
    dot.node('output', 'Attention(Q,K,V)\nWeighted Values', fillcolor=output_color)
    
    # Соединения
    dot.edge('Q', 'QK', label='×')
    dot.edge('K', 'QK', label='×')
    dot.edge('QK', 'scale', label='Scale')
    dot.edge('scale', 'fuzzy_mod', label='+')
    dot.edge('fuzzy_mod', 'softmax', label='Modulate')
    dot.edge('softmax', 'output', label='×')
    dot.edge('V', 'output', label='×')
    
    return dot

def create_fuzzy_functions_diagram():
    """Создает схему fuzzy membership functions"""
    
    dot = Digraph(comment='Fuzzy Membership Functions', format='png')
    dot.attr(rankdir='TB', size='8,6', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='11', shape='box', style='filled')
    
    # Цвета для разных функций
    colors = ['#FFCDD2', '#F8BBD9', '#E1BEE7', '#C5CAE9', '#BBDEFB', '#B3E5FC', '#B2EBF2']
    
    # Fuzzy functions
    functions = [
        ('f1', 'Very Low\nμ₁=0.1, σ₁=0.3'),
        ('f2', 'Low\nμ₂=0.3, σ₂=0.4'),
        ('f3', 'Medium\nμ₃=0.5, σ₃=0.5'),
        ('f4', 'High\nμ₄=0.7, σ₄=0.4'),
        ('f5', 'Very High\nμ₅=0.9, σ₅=0.3'),
        ('f6', 'Extreme\nμ₆=1.1, σ₆=0.2'),
        ('f7', 'Critical\nμ₇=1.3, σ₇=0.1')
    ]
    
    for i, (name, label) in enumerate(functions):
        dot.node(name, label, fillcolor=colors[i % len(colors)])
    
    # Input
    dot.node('input', 'Input Value\nx ∈ [0,1]', fillcolor='#E3F2FD', shape='ellipse')
    
    # Output
    dot.node('output', 'Membership Degrees\n[μ₁(x), μ₂(x), ..., μ₇(x)]', 
            fillcolor='#E0F2F1', shape='ellipse')
    
    # Соединения
    for name, _ in functions:
        dot.edge('input', name, label='μ(x)')
        dot.edge(name, 'output', label='')
    
    return dot

def main():
    """Генерирует все диаграммы"""
    
    print("🎨 Генерация схем архитектуры FAN для статьи уровня A...")
    
    # Создаем директорию для диаграмм
    os.makedirs('diagrams', exist_ok=True)
    
    # 1. Основная архитектура
    print("📊 Создание основной схемы архитектуры...")
    main_diagram = create_fan_architecture_diagram()
    main_diagram.render('diagrams/fan_architecture', cleanup=True)
    print("✅ Сохранено: diagrams/fan_architecture.png")
    
    # 2. Детальная схема attention head
    print("🔍 Создание детальной схемы attention head...")
    attention_diagram = create_attention_head_diagram()
    attention_diagram.render('diagrams/attention_head_detail', cleanup=True)
    print("✅ Сохранено: diagrams/attention_head_detail.png")
    
    # 3. Fuzzy membership functions
    print("🧠 Создание схемы fuzzy membership functions...")
    fuzzy_diagram = create_fuzzy_functions_diagram()
    fuzzy_diagram.render('diagrams/fuzzy_membership_functions', cleanup=True)
    print("✅ Сохранено: diagrams/fuzzy_membership_functions.png")
    
    print("\n🎉 Все диаграммы созданы успешно!")
    print("📁 Файлы сохранены в папке 'diagrams/'")
    print("\n📝 Для использования в статье:")
    print("   - fan_architecture.png - основная схема")
    print("   - attention_head_detail.png - детали attention")
    print("   - fuzzy_membership_functions.png - fuzzy функции")
    
    # Создаем также LaTeX код для вставки
    create_latex_code()

def create_latex_code():
    """Создает LaTeX код для вставки диаграмм в статью"""
    
    latex_code = """
% LaTeX код для вставки диаграмм FAN в статью
% Добавьте в преамбулу: \\usepackage{graphicx}

\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{diagrams/fan_architecture.png}
    \\caption{Архитектура Fuzzy Attention Networks (FAN). Система обрабатывает мультимодальные входы (текст и изображения) через специализированные энкодеры, применяет fuzzy attention механизм для модуляции attention weights, и выполняет мультимодальное слияние для финальной классификации.}
    \\label{fig:fan_architecture}
\\end{figure}

\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{diagrams/attention_head_detail.png}
    \\caption{Детальная схема fuzzy attention head. Attention weights модулируются fuzzy membership функциями для улучшения интерпретируемости и адаптивности модели.}
    \\label{fig:attention_detail}
\\end{figure}

\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.6\\textwidth]{diagrams/fuzzy_membership_functions.png}
    \\caption{Набор fuzzy membership функций, используемых для модуляции attention weights. Каждая функция представляет различную степень важности входных признаков.}
    \\label{fig:fuzzy_functions}
\\end{figure}
"""
    
    with open('diagrams/latex_code.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print("📝 LaTeX код сохранен: diagrams/latex_code.tex")

if __name__ == "__main__":
    main()



