#!/usr/bin/env python3
"""
Официальная схема архитектуры FAN для журнала уровня A
Строгий, профессиональный дизайн в академическом стиле
"""

def create_official_fan_architecture():
    """Создает официальную схему архитектуры FAN для журнала"""
    
    dot_content = """
digraph FAN_Architecture {
    rankdir=TB;
    size="10,12";
    dpi=300;
    
    // Настройки узлов - строгий академический стиль
    node [fontname="Times", fontsize=10, shape=box, style=filled, penwidth=1];
    edge [fontname="Times", fontsize=8, penwidth=1, color="#333333"];
    
    // Входные данные - нейтральные цвета
    text_input [label="Text Input\\n\\"Golden retriever dog\\"", fillcolor="#F5F5F5", shape=ellipse, fontcolor="#000000", penwidth=1.5];
    image_input [label="Image Input\\n224×224×3", fillcolor="#F5F5F5", shape=ellipse, fontcolor="#000000", penwidth=1.5];
    
    // Текстовый путь - серые тона
    bert_tokenizer [label="BERT Tokenizer", fillcolor="#E8E8E8", fontcolor="#000000"];
    bert_encoder [label="BERT Encoder\\n12 Layers", fillcolor="#D3D3D3", fontcolor="#000000"];
    text_projection [label="Text Projection\\nLinear(768→512)", fillcolor="#C0C0C0", fontcolor="#000000"];
    
    // Изображение путь - серые тона
    resnet_backbone [label="ResNet Backbone", fillcolor="#E8E8E8", fontcolor="#000000"];
    image_projection [label="Image Projection\\nLinear(2048→512)", fillcolor="#D3D3D3", fontcolor="#000000"];
    
    // Attention механизм - темно-серые тона
    query_proj [label="Query\\nLinear(512→512)", fillcolor="#A9A9A9", fontcolor="#FFFFFF"];
    key_proj [label="Key\\nLinear(512→512)", fillcolor="#A9A9A9", fontcolor="#FFFFFF"];
    value_proj [label="Value\\nLinear(512→512)", fillcolor="#A9A9A9", fontcolor="#FFFFFF"];
    
    // Fuzzy компоненты - черные тона
    fuzzy_centers [label="Fuzzy Centers\\nμ₁, μ₂, ..., μₖ", fillcolor="#696969", shape=diamond, fontcolor="#FFFFFF"];
    fuzzy_widths [label="Fuzzy Widths\\nσ₁, σ₂, ..., σₖ", fillcolor="#696969", shape=diamond, fontcolor="#FFFFFF"];
    bell_membership [label="Bell Membership\\nμ(x) = 1/(1+((x-μ)/σ)²)", fillcolor="#2F4F4F", shape=ellipse, fontcolor="#FFFFFF"];
    
    // Attention вычисления - темно-серые
    attention_weights [label="Attention Weights\\nSoftmax(QKᵀ/√d)", fillcolor="#808080", fontcolor="#FFFFFF"];
    attended_values [label="Attended Values\\nAttention × V", fillcolor="#696969", fontcolor="#FFFFFF"];
    
    // Fusion - серые тона
    concat_fusion [label="Concatenation\\n[text; image; attention]", fillcolor="#DCDCDC", fontcolor="#000000"];
    fusion_layer [label="Fusion Layer\\nLinear(1536→512)", fillcolor="#C0C0C0", fontcolor="#000000"];
    dropout [label="Dropout + ReLU", fillcolor="#A9A9A9", fontcolor="#FFFFFF"];
    
    // Классификатор - темно-серые
    classifier [label="Classifier\\nLinear(512→num_classes)", fillcolor="#708090", fontcolor="#FFFFFF"];
    softmax [label="Softmax", fillcolor="#556B2F", fontcolor="#FFFFFF"];
    prediction [label="Prediction\\nClass + Confidence", fillcolor="#2F4F4F", shape=ellipse, fontcolor="#FFFFFF"];
    
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
    
    // Математическая формула - строгий стиль
    formula [label="Attention(Q,K,V) = softmax(QK^T/√d + F(μ,σ))V", 
             fillcolor="#FFFFFF", shape=plaintext, fontsize=9, fontcolor="#000000", penwidth=1];
    
    // Соединяем формулу
    formula -> attention_weights [style=dashed, color="#666666"];
}
"""
    return dot_content

def main():
    """Генерирует официальную схему для журнала"""
    
    print("📚 Создание официальной схемы архитектуры FAN для журнала...")
    
    # Создаем директорию
    import os
    os.makedirs('diagrams', exist_ok=True)
    
    # Создаем официальную схему
    official_dot = create_official_fan_architecture()
    with open('diagrams/fan_architecture_official.dot', 'w', encoding='utf-8') as f:
        f.write(official_dot)
    
    print("✅ Сохранено: diagrams/fan_architecture_official.dot")
    
    # Создаем LaTeX код
    latex_code = """
% Официальная схема архитектуры FAN для журнала
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{diagrams/fan_architecture_official.png}
    \\caption{Архитектура Fuzzy Attention Networks (FAN). Система обрабатывает мультимодальные входы через BERT и ResNet энкодеры, применяет fuzzy attention механизм с bell membership функциями для модуляции attention weights, и выполняет мультимодальное слияние для финальной классификации.}
    \\label{fig:fan_architecture}
\\end{figure}
"""
    
    with open('diagrams/latex_official.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print("✅ Сохранено: diagrams/latex_official.tex")
    
    print("\n📚 Официальная схема для журнала создана!")
    print("📁 Файл: diagrams/fan_architecture_official.dot")
    print("📝 LaTeX: diagrams/latex_official.tex")
    
    print("\n🎨 Стиль (академический):")
    print("   ⚫ Черно-белая палитра с серыми тонами")
    print("   📰 Шрифт Times (академический стандарт)")
    print("   📐 Строгие геометрические формы")
    print("   🔲 Минималистичный дизайн")
    print("   📊 Профессиональное качество")
    
    print("\n🚀 Для конвертации в PNG:")
    print("   1. Откройте https://dreampuf.github.io/GraphvizOnline/")
    print("   2. Скопируйте содержимое fan_architecture_official.dot")
    print("   3. Вставьте и нажмите 'Generate'")
    print("   4. Скачайте PNG файл")
    
    print("\n📖 Готово для публикации в журнале уровня A!")

if __name__ == "__main__":
    main()



