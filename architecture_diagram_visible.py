#!/usr/bin/env python3
"""
Официальная схема архитектуры FAN с хорошей видимостью
Профессиональный дизайн для журнала с четкими контрастами
"""

def create_visible_fan_architecture():
    """Создает схему архитектуры FAN с хорошей видимостью"""
    
    dot_content = """
digraph FAN_Architecture {
    rankdir=TB;
    size="10,12";
    dpi=300;
    
    // Настройки узлов - четкие контрасты
    node [fontname="Arial", fontsize=10, shape=box, style=filled, penwidth=1.5];
    edge [fontname="Arial", fontsize=9, penwidth=1.5, color="#333333"];
    
    // Входные данные - светлые с темным текстом
    text_input [label="Text Input\\n\\"Golden retriever dog\\"", fillcolor="#E6F3FF", shape=ellipse, fontcolor="#000000", penwidth=2];
    image_input [label="Image Input\\n224×224×3", fillcolor="#E6FFE6", shape=ellipse, fontcolor="#000000", penwidth=2];
    
    // Текстовый путь - светлые серые
    bert_tokenizer [label="BERT Tokenizer", fillcolor="#F0F0F0", fontcolor="#000000"];
    bert_encoder [label="BERT Encoder\\n12 Layers", fillcolor="#E0E0E0", fontcolor="#000000"];
    text_projection [label="Text Projection\\nLinear(768→512)", fillcolor="#D0D0D0", fontcolor="#000000"];
    
    // Изображение путь - светлые серые
    resnet_backbone [label="ResNet Backbone", fillcolor="#F0F0F0", fontcolor="#000000"];
    image_projection [label="Image Projection\\nLinear(2048→512)", fillcolor="#E0E0E0", fontcolor="#000000"];
    
    // Attention механизм - средние серые
    query_proj [label="Query\\nLinear(512→512)", fillcolor="#B8B8B8", fontcolor="#000000"];
    key_proj [label="Key\\nLinear(512→512)", fillcolor="#B8B8B8", fontcolor="#000000"];
    value_proj [label="Value\\nLinear(512→512)", fillcolor="#B8B8B8", fontcolor="#000000"];
    
    // Fuzzy компоненты - темные с белым текстом
    fuzzy_centers [label="Fuzzy Centers\\nμ₁, μ₂, ..., μₖ", fillcolor="#606060", shape=diamond, fontcolor="#FFFFFF"];
    fuzzy_widths [label="Fuzzy Widths\\nσ₁, σ₂, ..., σₖ", fillcolor="#606060", shape=diamond, fontcolor="#FFFFFF"];
    bell_membership [label="Bell Membership\\nμ(x) = 1/(1+((x-μ)/σ)²)", fillcolor="#404040", shape=ellipse, fontcolor="#FFFFFF"];
    
    // Attention вычисления - средние серые
    attention_weights [label="Attention Weights\\nSoftmax(QKᵀ/√d)", fillcolor="#A0A0A0", fontcolor="#000000"];
    attended_values [label="Attended Values\\nAttention × V", fillcolor="#909090", fontcolor="#000000"];
    
    // Fusion - светлые серые
    concat_fusion [label="Concatenation\\n[text; image; attention]", fillcolor="#F5F5F5", fontcolor="#000000"];
    fusion_layer [label="Fusion Layer\\nLinear(1536→512)", fillcolor="#E8E8E8", fontcolor="#000000"];
    dropout [label="Dropout + ReLU", fillcolor="#D8D8D8", fontcolor="#000000"];
    
    // Классификатор - темные с белым текстом
    classifier [label="Classifier\\nLinear(512→num_classes)", fillcolor="#505050", fontcolor="#FFFFFF"];
    softmax [label="Softmax", fillcolor="#404040", fontcolor="#FFFFFF"];
    prediction [label="Prediction\\nClass + Confidence", fillcolor="#303030", shape=ellipse, fontcolor="#FFFFFF"];
    
    // Соединения - текстовый путь
    text_input -> bert_tokenizer [label="Tokenize", color="#000000"];
    bert_tokenizer -> bert_encoder [label="Encode", color="#000000"];
    bert_encoder -> text_projection [label="Project", color="#000000"];
    
    // Соединения - изображение путь
    image_input -> resnet_backbone [label="Extract", color="#000000"];
    resnet_backbone -> image_projection [label="Project", color="#000000"];
    
    // Соединения - attention механизм
    text_projection -> query_proj [label="Q", color="#000000"];
    text_projection -> key_proj [label="K", color="#000000"];
    text_projection -> value_proj [label="V", color="#000000"];
    
    fuzzy_centers -> bell_membership [label="μ", color="#000000"];
    fuzzy_widths -> bell_membership [label="σ", color="#000000"];
    
    query_proj -> attention_weights [label="Q", color="#000000"];
    key_proj -> attention_weights [label="K", color="#000000"];
    bell_membership -> attention_weights [label="Fuzzy Modulate", color="#000000"];
    
    attention_weights -> attended_values [label="Attention", color="#000000"];
    value_proj -> attended_values [label="V", color="#000000"];
    
    // Соединения - fusion
    text_projection -> concat_fusion [label="Text Features", color="#000000"];
    image_projection -> concat_fusion [label="Image Features", color="#000000"];
    attended_values -> concat_fusion [label="Attended Features", color="#000000"];
    
    concat_fusion -> fusion_layer [label="Fuse", color="#000000"];
    fusion_layer -> dropout [label="Process", color="#000000"];
    
    // Соединения - классификация
    dropout -> classifier [label="Classify", color="#000000"];
    classifier -> softmax [label="Normalize", color="#000000"];
    softmax -> prediction [label="Output", color="#000000"];
    
    // Математическая формула - четкий контраст
    formula [label="Attention(Q,K,V) = softmax(QK^T/√d + F(μ,σ))V", 
             fillcolor="#FFFFFF", shape=plaintext, fontsize=9, fontcolor="#000000", penwidth=1];
    
    // Соединяем формулу
    formula -> attention_weights [style=dashed, color="#666666"];
}
"""
    return dot_content

def main():
    """Генерирует схему с хорошей видимостью"""
    
    print("👁️ Создание схемы архитектуры FAN с хорошей видимостью...")
    
    # Создаем директорию
    import os
    os.makedirs('diagrams', exist_ok=True)
    
    # Создаем схему с хорошей видимостью
    visible_dot = create_visible_fan_architecture()
    with open('diagrams/fan_architecture_visible.dot', 'w', encoding='utf-8') as f:
        f.write(visible_dot)
    
    print("✅ Сохранено: diagrams/fan_architecture_visible.dot")
    
    # Создаем LaTeX код
    latex_code = """
% Схема архитектуры FAN с хорошей видимостью
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{diagrams/fan_architecture_visible.png}
    \\caption{Архитектура Fuzzy Attention Networks (FAN). Система обрабатывает мультимодальные входы через BERT и ResNet энкодеры, применяет fuzzy attention механизм с bell membership функциями для модуляции attention weights, и выполняет мультимодальное слияние для финальной классификации.}
    \\label{fig:fan_architecture}
\\end{figure}
"""
    
    with open('diagrams/latex_visible.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print("✅ Сохранено: diagrams/latex_visible.tex")
    
    print("\n👁️ Схема с хорошей видимостью создана!")
    print("📁 Файл: diagrams/fan_architecture_visible.dot")
    print("📝 LaTeX: diagrams/latex_visible.tex")
    
    print("\n🎨 Особенности (хорошая видимость):")
    print("   ⚪ Светлые фоны с темным текстом")
    print("   ⚫ Темные элементы с белым текстом")
    print("   📏 Четкие контрасты")
    print("   🔍 Хорошая читаемость")
    print("   📰 Академический стиль")
    print("   📊 Профессиональное качество")
    
    print("\n🚀 Для конвертации в PNG:")
    print("   1. Откройте https://dreampuf.github.io/GraphvizOnline/")
    print("   2. Скопируйте содержимое fan_architecture_visible.dot")
    print("   3. Вставьте и нажмите 'Generate'")
    print("   4. Скачайте PNG файл")
    
    print("\n📖 Готово для журнала с отличной видимостью!")

if __name__ == "__main__":
    main()



