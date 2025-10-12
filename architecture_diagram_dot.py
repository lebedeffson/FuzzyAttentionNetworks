#!/usr/bin/env python3
"""
Генератор DOT файлов для схемы архитектуры FAN
Создает .dot файлы, которые можно сконвертировать в PNG через онлайн конвертеры
"""

def create_fan_architecture_dot():
    """Создает DOT код для основной схемы архитектуры FAN"""
    
    dot_content = """
digraph FAN_Architecture {
    rankdir=TB;
    size="12,16";
    dpi=300;
    
    // Настройки узлов
    node [fontname="Arial", fontsize=12, shape=box, style=filled];
    edge [fontname="Arial", fontsize=10];
    
    // Цвета
    input_color [fillcolor="#E3F2FD", style=filled];
    text_color [fillcolor="#FFF3E0", style=filled];
    image_color [fillcolor="#E8F5E8", style=filled];
    attention_color [fillcolor="#F3E5F5", style=filled];
    fuzzy_color [fillcolor="#FFEBEE", style=filled];
    output_color [fillcolor="#E0F2F1", style=filled];
    fusion_color [fillcolor="#FFF8E1", style=filled];
    
    // Входные данные
    subgraph cluster_inputs {
        label="Input Layer";
        style=filled;
        color=lightgray;
        fontsize=14;
        fontname="Arial Bold";
        
        text_input [label="Text Input\\n\\"Golden retriever dog\\"", fillcolor="#E3F2FD", shape=ellipse];
        image_input [label="Image Input\\n224×224×3", fillcolor="#E3F2FD", shape=ellipse];
    }
    
    // Текстовый путь
    subgraph cluster_text {
        label="Text Processing Pipeline";
        style=filled;
        color=lightgray;
        fontsize=14;
        fontname="Arial Bold";
        
        bert_tokenizer [label="BERT Tokenizer\\n[CLS] golden retriever [SEP]", fillcolor="#FFF3E0"];
        bert_encoder [label="BERT Encoder\\n12 Layers, 768-dim", fillcolor="#FFF3E0"];
        text_projection [label="Text Projection\\nLinear(768→512)", fillcolor="#FFF3E0"];
    }
    
    // Изображение путь
    subgraph cluster_image {
        label="Image Processing Pipeline";
        style=filled;
        color=lightgray;
        fontsize=14;
        fontname="Arial Bold";
        
        resnet_backbone [label="ResNet Backbone\\nResNet50/ResNet18", fillcolor="#E8F5E8"];
        image_projection [label="Image Projection\\nLinear(2048→512)", fillcolor="#E8F5E8"];
    }
    
    // Fuzzy Attention механизм
    subgraph cluster_fuzzy_attention {
        label="Fuzzy Attention Mechanism";
        style=filled;
        color=lightgray;
        fontsize=14;
        fontname="Arial Bold";
        
        query_proj [label="Query Projection\\nLinear(512→512)", fillcolor="#F3E5F5"];
        key_proj [label="Key Projection\\nLinear(512→512)", fillcolor="#F3E5F5"];
        value_proj [label="Value Projection\\nLinear(512→512)", fillcolor="#F3E5F5"];
        
        fuzzy_centers [label="Fuzzy Centers\\nμ₁, μ₂, ..., μₖ", fillcolor="#FFEBEE", shape=diamond];
        fuzzy_widths [label="Fuzzy Widths\\nσ₁, σ₂, ..., σₖ", fillcolor="#FFEBEE", shape=diamond];
        
        bell_membership [label="Bell Membership\\nμ(x) = 1/(1+((x-μ)/σ)²)", fillcolor="#FFEBEE", shape=ellipse];
        
        attention_weights [label="Attention Weights\\nSoftmax(QKᵀ/√d)", fillcolor="#F3E5F5"];
        attended_values [label="Attended Values\\nAttention × V", fillcolor="#F3E5F5"];
    }
    
    // Мультимодальное слияние
    subgraph cluster_fusion {
        label="Multimodal Fusion";
        style=filled;
        color=lightgray;
        fontsize=14;
        fontname="Arial Bold";
        
        concat_fusion [label="Concatenation\\n[text; image; attention]", fillcolor="#FFF8E1"];
        fusion_layer [label="Fusion Layer\\nLinear(1536→512)", fillcolor="#FFF8E1"];
        dropout [label="Dropout\\np=0.1", fillcolor="#FFF8E1"];
        relu [label="ReLU Activation", fillcolor="#FFF8E1"];
    }
    
    // Классификатор
    subgraph cluster_classifier {
        label="Classification Head";
        style=filled;
        color=lightgray;
        fontsize=14;
        fontname="Arial Bold";
        
        classifier [label="Classifier\\nLinear(512→num_classes)", fillcolor="#E0F2F1"];
        softmax [label="Softmax\\nProbability Distribution", fillcolor="#E0F2F1"];
        prediction [label="Prediction\\nClass + Confidence", fillcolor="#E0F2F1", shape=ellipse];
    }
    
    // Соединения - текстовый путь
    text_input -> bert_tokenizer [label="Tokenize"];
    bert_tokenizer -> bert_encoder [label="Encode"];
    bert_encoder -> text_projection [label="Project"];
    
    // Соединения - изображение путь
    image_input -> resnet_backbone [label="Extract Features"];
    resnet_backbone -> image_projection [label="Project"];
    
    // Соединения - attention механизм
    text_projection -> query_proj [label="Q"];
    text_projection -> key_proj [label="K"];
    text_projection -> value_proj [label="V"];
    
    fuzzy_centers -> bell_membership [label="μ"];
    fuzzy_widths -> bell_membership [label="σ"];
    query_proj -> attention_weights [label="Q"];
    key_proj -> attention_weights [label="K"];
    bell_membership -> attention_weights [label="Fuzzy Modulate"];
    attention_weights -> attended_values [label="Attention"];
    value_proj -> attended_values [label="V"];
    
    // Соединения - fusion
    text_projection -> concat_fusion [label="Text Features"];
    image_projection -> concat_fusion [label="Image Features"];
    attended_values -> concat_fusion [label="Attended Features"];
    
    concat_fusion -> fusion_layer [label="Fuse"];
    fusion_layer -> dropout [label="Regularize"];
    dropout -> relu [label="Activate"];
    
    // Соединения - классификация
    relu -> classifier [label="Classify"];
    classifier -> softmax [label="Normalize"];
    softmax -> prediction [label="Output"];
    
    // Математические формулы
    formula1 [label="Attention(Q,K,V) = softmax(QK^T/√d + F(μ,σ))V", fillcolor=white, shape=plaintext, fontsize=10];
    formula2 [label="F(μ,σ) = Σᵢ 1/(1+((x-μᵢ)/σᵢ)²)", fillcolor=white, shape=plaintext, fontsize=10];
    
    // Соединяем формулы
    formula1 -> attention_weights [style=dashed, color=gray];
    formula2 -> bell_membership [style=dashed, color=gray];
}
"""
    return dot_content

def create_attention_head_dot():
    """Создает DOT код для детальной схемы attention head"""
    
    dot_content = """
digraph AttentionHeadDetail {
    rankdir=LR;
    size="10,8";
    dpi=300;
    
    node [fontname="Arial", fontsize=11, shape=box, style=filled];
    
    // Входы
    Q [label="Query\\nQ ∈ R^(d×d)", fillcolor="#E3F2FD"];
    K [label="Key\\nK ∈ R^(d×d)", fillcolor="#E3F2FD"];
    V [label="Value\\nV ∈ R^(d×d)", fillcolor="#E3F2FD"];
    
    // Attention computation
    QK [label="QK^T\\nAttention Scores", fillcolor="#F3E5F5"];
    scale [label="Scale\\n÷√d", fillcolor="#F3E5F5"];
    
    // Fuzzy modulation
    fuzzy_mod [label="Fuzzy Modulation\\nF(μ,σ)", fillcolor="#FFEBEE"];
    softmax [label="Softmax\\nNormalization", fillcolor="#F3E5F5"];
    
    // Output
    output [label="Attention(Q,K,V)\\nWeighted Values", fillcolor="#E0F2F1"];
    
    // Соединения
    Q -> QK [label="×"];
    K -> QK [label="×"];
    QK -> scale [label="Scale"];
    scale -> fuzzy_mod [label="+"];
    fuzzy_mod -> softmax [label="Modulate"];
    softmax -> output [label="×"];
    V -> output [label="×"];
}
"""
    return dot_content

def create_fuzzy_functions_dot():
    """Создает DOT код для схемы fuzzy membership functions"""
    
    dot_content = """
digraph FuzzyMembershipFunctions {
    rankdir=TB;
    size="8,6";
    dpi=300;
    
    node [fontname="Arial", fontsize=11, shape=box, style=filled];
    
    // Input
    input [label="Input Value\\nx ∈ [0,1]", fillcolor="#E3F2FD", shape=ellipse];
    
    // Fuzzy functions
    f1 [label="Very Low\\nμ₁=0.1, σ₁=0.3", fillcolor="#FFCDD2"];
    f2 [label="Low\\nμ₂=0.3, σ₂=0.4", fillcolor="#F8BBD9"];
    f3 [label="Medium\\nμ₃=0.5, σ₃=0.5", fillcolor="#E1BEE7"];
    f4 [label="High\\nμ₄=0.7, σ₄=0.4", fillcolor="#C5CAE9"];
    f5 [label="Very High\\nμ₅=0.9, σ₅=0.3", fillcolor="#BBDEFB"];
    f6 [label="Extreme\\nμ₆=1.1, σ₆=0.2", fillcolor="#B3E5FC"];
    f7 [label="Critical\\nμ₇=1.3, σ₇=0.1", fillcolor="#B2EBF2"];
    
    // Output
    output [label="Membership Degrees\\n[μ₁(x), μ₂(x), ..., μ₇(x)]", fillcolor="#E0F2F1", shape=ellipse];
    
    // Соединения
    input -> f1 [label="μ(x)"];
    input -> f2 [label="μ(x)"];
    input -> f3 [label="μ(x)"];
    input -> f4 [label="μ(x)"];
    input -> f5 [label="μ(x)"];
    input -> f6 [label="μ(x)"];
    input -> f7 [label="μ(x)"];
    
    f1 -> output [label=""];
    f2 -> output [label=""];
    f3 -> output [label=""];
    f4 -> output [label=""];
    f5 -> output [label=""];
    f6 -> output [label=""];
    f7 -> output [label=""];
}
"""
    return dot_content

def main():
    """Генерирует все DOT файлы"""
    
    print("🎨 Генерация DOT файлов для схем архитектуры FAN...")
    
    # Создаем директорию для диаграмм
    import os
    os.makedirs('diagrams', exist_ok=True)
    
    # 1. Основная архитектура
    print("📊 Создание основной схемы архитектуры...")
    main_dot = create_fan_architecture_dot()
    with open('diagrams/fan_architecture.dot', 'w', encoding='utf-8') as f:
        f.write(main_dot)
    print("✅ Сохранено: diagrams/fan_architecture.dot")
    
    # 2. Детальная схема attention head
    print("🔍 Создание детальной схемы attention head...")
    attention_dot = create_attention_head_dot()
    with open('diagrams/attention_head_detail.dot', 'w', encoding='utf-8') as f:
        f.write(attention_dot)
    print("✅ Сохранено: diagrams/attention_head_detail.dot")
    
    # 3. Fuzzy membership functions
    print("🧠 Создание схемы fuzzy membership functions...")
    fuzzy_dot = create_fuzzy_functions_dot()
    with open('diagrams/fuzzy_membership_functions.dot', 'w', encoding='utf-8') as f:
        f.write(fuzzy_dot)
    print("✅ Сохранено: diagrams/fuzzy_membership_functions.dot")
    
    print("\n🎉 Все DOT файлы созданы успешно!")
    print("📁 Файлы сохранены в папке 'diagrams/'")
    print("\n📝 Для конвертации в PNG:")
    print("   1. Откройте https://dreampuf.github.io/GraphvizOnline/")
    print("   2. Скопируйте содержимое .dot файла")
    print("   3. Вставьте в редактор и нажмите 'Generate'")
    print("   4. Скачайте PNG файл")
    
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

% Альтернативно, можно использовать TikZ для создания диаграмм прямо в LaTeX
\\usepackage{tikz}
\\usetikzlibrary{shapes,arrows,positioning}

\\begin{figure}[htbp]
    \\centering
    \\begin{tikzpicture}[node distance=2cm, auto]
        % Здесь можно добавить TikZ код для создания диаграммы
        \\node (input) {Input};
        \\node (output) [right of=input] {Output};
        \\draw [->] (input) -- (output);
    \\end{tikzpicture}
    \\caption{Пример TikZ диаграммы}
    \\label{fig:tikz_example}
\\end{figure}
"""
    
    with open('diagrams/latex_code.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print("📝 LaTeX код сохранен: diagrams/latex_code.tex")

if __name__ == "__main__":
    main()



