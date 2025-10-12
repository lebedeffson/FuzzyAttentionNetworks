#!/usr/bin/env python3
"""
Финальная схема архитектуры FAN с нежными, мягкими цветами
Профессиональный вид для статьи уровня A
"""

def create_soft_fan_architecture():
    """Создает схему архитектуры FAN с мягкими цветами"""
    
    dot_content = """
digraph FAN_Architecture {
    rankdir=TB;
    size="10,12";
    dpi=300;
    
    // Настройки узлов
    node [fontname="Arial", fontsize=11, shape=box, style=filled, penwidth=1.5];
    edge [fontname="Arial", fontsize=9, penwidth=1.5, color="#666666"];
    
    // Входные данные - мягкие пастельные
    text_input [label="Text Input\\n\\"Golden retriever dog\\"", fillcolor="#E8F4FD", shape=ellipse, fontcolor="#2C3E50", penwidth=2];
    image_input [label="Image Input\\n224×224×3", fillcolor="#E8F8F5", shape=ellipse, fontcolor="#2C3E50", penwidth=2];
    
    // Текстовый путь - нежные оранжевые
    bert_tokenizer [label="BERT Tokenizer", fillcolor="#FFF2E6", fontcolor="#8B4513"];
    bert_encoder [label="BERT Encoder\\n12 Layers", fillcolor="#FFE4CC", fontcolor="#8B4513"];
    text_projection [label="Text Projection\\n768→512", fillcolor="#FFD9B3", fontcolor="#8B4513"];
    
    // Изображение путь - мягкие зеленые
    resnet_backbone [label="ResNet Backbone", fillcolor="#E8F5E8", fontcolor="#2E7D32"];
    image_projection [label="Image Projection\\n2048→512", fillcolor="#D4EDDA", fontcolor="#2E7D32"];
    
    // Attention механизм - нежные фиолетовые
    query_proj [label="Query\\nLinear(512→512)", fillcolor="#F3E5F5", fontcolor="#6A1B9A"];
    key_proj [label="Key\\nLinear(512→512)", fillcolor="#E1BEE7", fontcolor="#6A1B9A"];
    value_proj [label="Value\\nLinear(512→512)", fillcolor="#CE93D8", fontcolor="#6A1B9A"];
    
    // Fuzzy компоненты - мягкие розовые
    fuzzy_centers [label="Fuzzy Centers\\nμ₁, μ₂, ..., μₖ", fillcolor="#FCE4EC", shape=diamond, fontcolor="#C2185B"];
    fuzzy_widths [label="Fuzzy Widths\\nσ₁, σ₂, ..., σₖ", fillcolor="#F8BBD9", shape=diamond, fontcolor="#C2185B"];
    bell_membership [label="Bell Membership\\nμ(x) = 1/(1+((x-μ)/σ)²)", fillcolor="#F48FB1", shape=ellipse, fontcolor="#C2185B"];
    
    // Attention вычисления - нежные синие
    attention_weights [label="Attention Weights\\nSoftmax(QKᵀ/√d)", fillcolor="#E3F2FD", fontcolor="#1565C0"];
    attended_values [label="Attended Values\\nAttention × V", fillcolor="#BBDEFB", fontcolor="#1565C0"];
    
    // Fusion - мягкие желтые
    concat_fusion [label="Concatenation\\n[text; image; attention]", fillcolor="#FFFDE7", fontcolor="#F57F17"];
    fusion_layer [label="Fusion Layer\\nLinear(1536→512)", fillcolor="#FFF9C4", fontcolor="#F57F17"];
    dropout [label="Dropout + ReLU", fillcolor="#FFF59D", fontcolor="#F57F17"];
    
    // Классификатор - нежные бирюзовые
    classifier [label="Classifier\\nLinear(512→classes)", fillcolor="#E0F2F1", fontcolor="#00695C"];
    softmax [label="Softmax", fillcolor="#B2DFDB", fontcolor="#00695C"];
    prediction [label="Prediction\\nClass + Confidence", fillcolor="#80CBC4", shape=ellipse, fontcolor="#00695C"];
    
    // Соединения - текстовый путь
    text_input -> bert_tokenizer [label="Tokenize", color="#3498DB"];
    bert_tokenizer -> bert_encoder [label="Encode", color="#E67E22"];
    bert_encoder -> text_projection [label="Project", color="#D35400"];
    
    // Соединения - изображение путь
    image_input -> resnet_backbone [label="Extract", color="#27AE60"];
    resnet_backbone -> image_projection [label="Project", color="#229954"];
    
    // Соединения - attention механизм
    text_projection -> query_proj [label="Q", color="#D35400"];
    text_projection -> key_proj [label="K", color="#D35400"];
    text_projection -> value_proj [label="V", color="#D35400"];
    
    fuzzy_centers -> bell_membership [label="μ", color="#E91E63"];
    fuzzy_widths -> bell_membership [label="σ", color="#C2185B"];
    
    query_proj -> attention_weights [label="Q", color="#8E24AA"];
    key_proj -> attention_weights [label="K", color="#7B1FA2"];
    bell_membership -> attention_weights [label="Fuzzy", color="#AD1457"];
    
    attention_weights -> attended_values [label="Attend", color="#1976D2"];
    value_proj -> attended_values [label="V", color="#6A1B9A"];
    
    // Соединения - fusion
    text_projection -> concat_fusion [label="Text", color="#D35400"];
    image_projection -> concat_fusion [label="Image", color="#229954"];
    attended_values -> concat_fusion [label="Attention", color="#1976D2"];
    
    concat_fusion -> fusion_layer [label="Fuse", color="#F39C12"];
    fusion_layer -> dropout [label="Process", color="#E67E22"];
    
    // Соединения - классификация
    dropout -> classifier [label="Classify", color="#F39C12"];
    classifier -> softmax [label="Normalize", color="#00ACC1"];
    softmax -> prediction [label="Output", color="#0097A7"];
    
    // Математическая формула - нежный серый
    formula [label="Attention(Q,K,V) = softmax(QK^T/√d + F(μ,σ))V", 
             fillcolor="#F8F9FA", shape=plaintext, fontsize=10, fontcolor="#495057", penwidth=1];
    
    // Соединяем формулу
    formula -> attention_weights [style=dashed, color="#6C757D"];
}
"""
    return dot_content

def main():
    """Генерирует финальную схему с мягкими цветами"""
    
    print("🎨 Создание финальной схемы с нежными, мягкими цветами...")
    
    # Создаем директорию
    import os
    os.makedirs('diagrams', exist_ok=True)
    
    # Создаем финальную схему
    final_dot = create_soft_fan_architecture()
    with open('diagrams/fan_architecture_soft.dot', 'w', encoding='utf-8') as f:
        f.write(final_dot)
    
    print("✅ Сохранено: diagrams/fan_architecture_soft.dot")
    
    # Создаем LaTeX код
    latex_code = """
% Финальная схема архитектуры FAN с мягкими цветами
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{diagrams/fan_architecture_soft.png}
    \\caption{Архитектура Fuzzy Attention Networks (FAN). Система обрабатывает мультимодальные входы через BERT и ResNet энкодеры, применяет fuzzy attention механизм с bell membership функциями для модуляции attention weights, и выполняет мультимодальное слияние для финальной классификации.}
    \\label{fig:fan_architecture}
\\end{figure}
"""
    
    with open('diagrams/latex_soft.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print("✅ Сохранено: diagrams/latex_soft.tex")
    
    print("\n🎉 Финальная схема с мягкими цветами создана!")
    print("📁 Файл: diagrams/fan_architecture_soft.dot")
    print("📝 LaTeX: diagrams/latex_soft.tex")
    
    print("\n🎨 Цветовая палитра (нежные тона):")
    print("   💙 Голубой - Входы")
    print("   🧡 Персиковый - Текстовый путь")
    print("   💚 Мятный - Изображение путь")
    print("   💜 Лавандовый - Attention механизм")
    print("   🩷 Розовый - Fuzzy компоненты")
    print("   💙 Небесный - Attention вычисления")
    print("   💛 Кремовый - Fusion")
    print("   💚 Морской - Классификация")
    
    print("\n🚀 Для конвертации в PNG:")
    print("   1. Откройте https://dreampuf.github.io/GraphvizOnline/")
    print("   2. Скопируйте содержимое fan_architecture_soft.dot")
    print("   3. Вставьте и нажмите 'Generate'")
    print("   4. Скачайте PNG файл")

if __name__ == "__main__":
    main()



