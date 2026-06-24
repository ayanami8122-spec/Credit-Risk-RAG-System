import pandas as pd
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def generate_performance_bars():
    df = pd.read_csv("rag_eval_results.csv")
    # 去除无效 0 分和 NaN
    df_scored = df[(df['faithfulness'] > 0) & (df['answer_relevancy'] > 0)]
    
    metrics = {
        '检索召回率 (Context Recall)': df_scored['context_recall'].mean(),
        '检索精准度 (Context Precision)': df_scored['context_precision'].mean(),
        '回答相关性 (Answer Relevance)': df_scored['answer_relevancy'].mean(),
        '回答忠实度 (Faithfulness)': df_scored['faithfulness'].mean()
    }
    
    # 绘图
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#1a5276', '#2980b9', '#5499c7', '#a9cce3'] # 深蓝色系渐变

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, values, color=colors, height=0.6)
    
    # 在条形图末尾标注数值
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', va='center', fontsize=14, fontweight='bold', color='#1a5276')

    plt.xlim(0, 1.1)
    plt.title('金融风控 RAG 系统 - 性能看板', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('得分 (0.0 - 1.0)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("performance_dashboard.png", dpi=300)
    print("✅ 更加清晰的性能看板已保存为: performance_dashboard.png")
    plt.show()

generate_performance_bars()