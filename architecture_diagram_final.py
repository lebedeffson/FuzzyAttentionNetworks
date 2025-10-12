#!/usr/bin/env python3
"""
Финальная красивая схема архитектуры FAN для статьи
Одна диаграмма с красивыми цветами, без серых блоков
"""

def create_final_fan_architecture():
    """Создает финальную красивую схему архитектуры FAN"""
    
    dot_content = """
digraph FAN_Architecture {
    rankdir=TB;
    size="10,12";
    dpi=300;
    
    // Настройки узлов
    node [fontname="Arial", fontsize=11, shape=box, style=filled, penwidth=2];
    edge [fontname="Arial", fontsize=9, penwidth=2];
    
    // Входные данные - яркие цвета
    text_input [label="Text Input\\n\\"Golden retriever dog\\"", fillcolor="#FF6B6B", shape=ellipse, fontcolor=white];
    image_input [label="Image Input\\n224×224×3", fillcolor="#4ECDC4", shape=ellipse, fontcolor=white];
    
    // Текстовый путь - оранжевые тона
    bert_tokenizer [label="BERT Tokenizer", fillcolor="#FFB74D", fontcolor=white];
    bert_encoder [label="BERT Encoder\\n12 Layers", fillcolor="#FF8A65", fontcolor=white];
    text_projection [label="Text Projection\\n768→512", fillcolor="#FF7043", fontcolor=white];
    
    // Изображение путь - зеленые тона
    resnet_backbone [label="ResNet Backbone", fillcolor="#81C784", fontcolor=white];
    image_projection [label="Image Projection\\n2048→512", fillcolor="#66BB6A", fontcolor=white];
    
    // Attention механизм - фиолетовые тона
    query_proj [label="Query\\nLinear(512→512)", fillcolor="#BA68C8", fontcolor=white];
    key_proj [label="Key\\nLinear(512→512)", fillcolor="#AB47BC", fontcolor=white];
    value_proj [label="Value\\nLinear(512→512)", fillcolor="#9C27B0", fontcolor=white];
    
    // Fuzzy компоненты - розовые тона
    fuzzy_centers [label="Fuzzy Centers\\nμ₁, μ₂, ..., μₖ", fillcolor="#F06292", shape=diamond, fontcolor=white];
    fuzzy_widths [label="Fuzzy Widths\\nσ₁, σ₂, ..., σₖ", fillcolor="#EC407A", shape=diamond, fontcolor=white];
    bell_membership [label="Bell Membership\\nμ(x) = 1/(1+((x-μ)/σ)²)", fillcolor="#E91E63", shape=ellipse, fontcolor=white];
    
    // Attention вычисления - синие тона
    attention_weights [label="Attention Weights\\nSoftmax(QKᵀ/√d)", fillcolor="#42A5F5", fontcolor=white];
    attended_values [label="Attended Values\\nAttention × V", fillcolor="#2196F3", fontcolor=white];
    
    // Fusion - желтые тона
    concat_fusion [label="Concatenation\\n[text; image; attention]", fillcolor="#FFD54F", fontcolor=black];
    fusion_layer [label="Fusion Layer\\nLinear(1536→512)", fillcolor="#FFC107", fontcolor=black];
    dropout [label="Dropout + ReLU", fillcolor="#FFB300", fontcolor=black];
    
    // Классификатор - бирюзовые тона
    classifier [label="Classifier\\nLinear(512→classes)", fillcolor="#26C6DA", fontcolor=white];
    softmax [label="Softmax", fillcolor="#00BCD4", fontcolor=white];
    prediction [label="Prediction\\nClass + Confidence", fillcolor="#00ACC1", shape=ellipse, fontcolor=white];
    
    // Соединения - текстовый путь
    text_input -> bert_tokenizer [label="Tokenize", color="#FF6B6B"];
    bert_tokenizer -> bert_encoder [label="Encode", color="#FF8A65"];
    bert_encoder -> text_projection [label="Project", color="#FF7043"];
    
    // Соединения - изображение путь
    image_input -> resnet_backbone [label="Extract", color="#4ECDC4"];
    resnet_backbone -> image_projection [label="Project", color="#66BB6A"];
    
    // Соединения - attention механизм
    text_projection -> query_proj [label="Q", color="#FF7043"];
    text_projection -> key_proj [label="K", color="#FF7043"];
    text_projection -> value_proj [label="V", color="#FF7043"];
    
    fuzzy_centers -> bell_membership [label="μ", color="#F06292"];
    fuzzy_widths -> bell_membership [label="σ", color="#EC407A"];
    
    query_proj -> attention_weights [label="Q", color="#BA68C8"];
    key_proj -> attention_weights [label="K", color="#AB47BC"];
    bell_membership -> attention_weights [label="Fuzzy", color="#E91E63"];
    
    attention_weights -> attended_values [label="Attend", color="#42A5F5"];
    value_proj -> attended_values [label="V", color="#9C27B0"];
    
    // Соединения - fusion
    text_projection -> concat_fusion [label="Text", color="#FF7043"];
    image_projection -> concat_fusion [label="Image", color="#66BB6A"];
    attended_values -> concat_fusion [label="Attention", color="#2196F3"];
    
    concat_fusion -> fusion_layer [label="Fuse", color="#FFD54F"];
    fusion_layer -> dropout [label="Process", color="#FFC107"];
    
    // Соединения - классификация
    dropout -> classifier [label="Classify", color="#FFB300"];
    classifier -> softmax [label="Normalize", color="#26C6DA"];
    softmax -> prediction [label="Output", color="#00BCD4"];
    
    // Математическая формула
    formula [label="Attention(Q,K,V) = softmax(QK^T/√d + F(μ,σ))V", 
             fillcolor=white, shape=plaintext, fontsize=10, fontcolor="#333333"];
    
    // Соединяем формулу
    formula -> attention_weights [style=dashed, color=gray];
}
"""
    return dot_content

def main():
    """Генерирует финальную схему"""
    
    print("🎨 Создание финальной красивой схемы архитектуры FAN...")
    
    # Создаем директорию
    import os
    os.makedirs('diagrams', exist_ok=True)
    
    # Создаем финальную схему
    final_dot = create_final_fan_architecture()
    with open('diagrams/fan_architecture_final.dot', 'w', encoding='utf-8') as f:
        f.write(final_dot)
    
    print("✅ Сохранено: diagrams/fan_architecture_final.dot")
    
    # Создаем LaTeX код
    latex_code = """
% Финальная схема архитектуры FAN для статьи
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{diagrams/fan_architecture_final.png}
    \\caption{Архитектура Fuzzy Attention Networks (FAN). Система обрабатывает мультимодальные входы через BERT и ResNet энкодеры, применяет fuzzy attention механизм с bell membership функциями для модуляции attention weights, и выполняет мультимодальное слияние для финальной классификации.}
    \\label{fig:fan_architecture}
\\end{figure}
"""
    
    with open('diagrams/latex_final.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print("✅ Сохранено: diagrams/latex_final.tex")
    
    print("\n🎉 Финальная схема создана!")
    print("📁 Файл: diagrams/fan_architecture_final.dot")
    print("📝 LaTeX: diagrams/latex_final.tex")
    print("\n🚀 Для конвертации в PNG:")
    print("   1. Откройте https://dreampuf.github.io/GraphvizOnline/")
    print("   2. Скопируйте содержимое fan_architecture_final.dot")
    print("   3. Вставьте и нажмите 'Generate'")
    print("   4. Скачайте PNG файл")
    
    print("\n🎨 Цветовая схема:")
    print("   🔴 Красный - Входы")
    print("   🟠 Оранжевый - Текстовый путь")
    print("   🟢 Зеленый - Изображение путь")
    print("   🟣 Фиолетовый - Attention механизм")
    print("   🩷 Розовый - Fuzzy компоненты")
    print("   🔵 Синий - Attention вычисления")
    print("   🟡 Желтый - Fusion")
    print("   🔷 Бирюзовый - Классификация")

if __name__ == "__main__":
    main()



