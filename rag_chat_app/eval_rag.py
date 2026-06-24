import os
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
import warnings
from ragas import RunConfig

# ragas 0.4+：正确类名为 AnswerRelevancy（不是 AnswerRelevance）
# 与 evaluate() + LangChain LLM 兼容的是 legacy 指标（collections 需 Instructor LLM）
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from typing import Any
from pydantic import SecretStr
from rag_pipeline import get_rag_chain
import traceback
from langchain_huggingface import HuggingFaceEmbeddings

faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy()
context_precision = ContextPrecision()
context_recall = ContextRecall()

# 1. 准备“黄金数据集” (Golden Dataset)
test_questions = [

    {
        "question": "现代信用风险的定义与传统定义相比，核心变化是什么？",
        "ground_truth": "传统定义主要聚焦于“违约风险”，即只有在违约事件发生时才确认损失。现代定义则演变为“由于债务人信用质量恶化或评级下降导致金融资产市场价值下降的可能性”，强调即使未实质违约，只要偿债预期变弱，资产价值就会发生动态减损。"
    },

    {
        "question": "什么是错向风险（Wrong-Way Risk, WWR）？请简述其机理。",
        "ground_truth": "错向风险是指对手方的违约概率（PD）与交易的风险敞口（EAD）之间存在正相关性。其机理在于：当对手方最可能发生违约时，债权人面临的损失规模（EAD）也恰好处于最高峰。这种“PD 与 EAD 共振”的现象会显著放大信用损失。"
    },

    {
        "question": "巴塞尔协议 IV (Basel IV) 引入的“输出底线 (Output Floor)”具体指什么？其目的是什么？",
        "ground_truth": "输出底线规定银行使用内部模型法（IRB）计算得到的风险加权资产（RWA）总额，不得低于使用监管标准法（SA）计算结果的 72.5%。其目的是限制银行利用复杂模型过度调低风险权重，防止资本计提不足，提高监管透明度。"
    },

    {
        "question": "宏观审慎框架中，逆周期资本缓冲（CCyB）是如何应对信用风险顺周期性的？",
        "ground_truth": "逆周期资本缓冲是一种对冲工具。当信贷增速远超 GDP 增速（繁荣期）时，监管机构要求银行额外计提资本（0%-2.5%）；当经济进入衰退期时，释放该缓冲资本，以鼓励银行继续支持实体经济，从而平滑信贷周期中的波动。"
    },

    {
        "question": "在违约概率度量中，时点 PD（PIT）与跨周期 PD（TTC）的主要区别是什么？",
        "ground_truth": "时点 PD（PIT）反映当前宏观经济环境下的即时违约概率，其数值随经济波动较为剧烈；而跨周期 PD（TTC）反映债务人在整个经济周期内的平均违约倾向，具有更强的稳定性，受短期经济波动影响较小。"
    },

    {
        "question": "简述 Merton 模型中“违约即期权”的逻辑以及违约距离（DD）的含义。",
        "ground_truth": "Merton 模型认为企业股权本质上是以公司资产价值为标的、以债务本息为执行价格的看涨期权；当资产价值低于债务时，股东会放弃期权选择违约。违约距离（DD）则代表公司资产价值距离债务警戒线的标准差倍数，DD 越大，代表违约概率越低。"
    },

    {
        "question": "在传统评分卡建模中，IV（信息价值）的作用是什么？通常什么数值代表特征具有极强预测力？",
        "ground_truth": "IV（Information Value）用于筛选特征变量，衡量每个变量对违约目标的预测能力。在实务中，通常认为 IV 值大于 0.3 的变量具有极强的预测动力。"
    },

    {
        "question": "什么是群体稳定性指数（PSI）？当 PSI 超过多少时通常需要重新训练模型？",
        "ground_truth": "PSI 用于监控实际进件人群与建模人群的分布差异，衡量模型的稳定性。如果 PSI 指数大于 0.25，则通常认为模型分布发生了显著偏移，需要重新训练或进行模型迭代。"
    },

    {
        "question": "什么是经济违约损失率（Economic LGD）？它与会计损失的主要区别是什么？",
        "ground_truth": "经济 LGD 要求在计算损失时包含货币的时间价值以及清收过程中的直接与间接成本，即对回收的现金流进行折现。与侧重于账面本金减值的会计损失不同，即使本金全部收回，若由于清收周期过长导致现值（PV）减少或消耗了法律评估成本，在经济 LGD 视角下依然构成实质性损失。"
    },

    {
        "question": "为什么传统的线性回归（OLS）在预测 LGD 时往往失效？LGD 的概率分布通常具有什么特征？",
        "ground_truth": "因为 LGD 的分布通常呈现典型的“双峰”特征，即债务人违约后，债权人要么能回收大部分资产，要么几乎全额损失，这导致均值回归的线性模型难以准确捕捉极值情况。在建模时，通常改用 Beta 回归等方法来模拟其在 [0, 1] 区间内的分布。"
    },

    {
        "question": "在经济衰退期，PD（违约概率）与 LGD（违约损失率）之间存在怎样的关系？这种现象被称为什么？",
        "ground_truth": "在经济衰退期，PD 与 LGD 存在显著的正相关性（共振效应）。当宏观经济恶化导致违约企业激增（PD 升高）时，市场上待处置抵押品供应过剩导致资产价格下跌，从而使得回收率下降、损失深度增加（LGD 升高）。这种“PD 与 LGD 同步升高”的现象是系统性风险的核心来源。"
    },

    {
        "question": "为什么说 EAD 是一个带有随机性的动态变量？它通常由哪几部分组成？",
        "ground_truth": "EAD 具有随机性是因为对于循环授信（如信用卡）或金融衍生品，违约发生时的敞口取决于债务人的行为（如紧急提领额度）及市场波动。它通常由三部分组成：已提取金额、未提取金额的预计提领部分（Commitment）、以及截至违约日累计产生的未付利息与追索费用。"
    },

    {
        "question": "什么是信用转换系数（CCF）？其背后的核心风控逻辑是什么？",
        "ground_truth": "CCF 用于将表外项目（如保函、未使用额度）转化为等值的表内风险敞口。其背后的核心逻辑是：当企业陷入财务困境时，往往会倾向于在违约前“突击提用”所有可用的信用额度以自救，因此违约时刻的真实敞口通常远高于观察时刻的余额。"
    },

    {
        "question": "在计量金融衍生品的 EAD 时，重置成本（RC）与潜在未来风险暴露（PFE）分别代表什么？",
        "ground_truth": "重置成本（RC）即当前市值（MtM），代表如果对手方今日违约，银行重建相同头寸所需的成本（仅当 MtM > 0 时计入）。潜在未来风险暴露（PFE）衡量在合同剩余期限内，由于市场波动可能导致的市值增加，通常由名义本金乘以附加因子得出，受期限和底层资产波动率影响。"
    },

    {
        "question": "在风险管理中，预期损失（EL）与非预期损失（UL）在财务处理上有什么本质区别？",
        "ground_truth": "预期损失（EL）被视为“做生意的成本”，应通过贷款风险定价来覆盖，并从利润中计提拨备。而非预期损失（UL）代表损失分布中偏离均值的突发部分，具有不可预测性，必须由银行的资本金（Capital）来抵御。"
    },

    {
        "question": "相比于旧准则（IAS 39），IFRS 9 减值模型最大的改进是什么？",
        "ground_truth": "最大的改进是从“已发生损失模型”转向“预期信用损失（ECL）模型”。它不再等到信用事件发生（如逾期）才计提减值，而是要求在资产初始确认时就基于前瞻性信息计提减值，有效解决了减值计提“太少、太迟”的问题。"
    },

    {
        "question": "IFRS 9 下的 ECL 测算如何体现“前瞻性（Forward-looking）”？",
        "ground_truth": "ECL 的测算要求使用概率加权的多个宏观场景。银行必须设定基准、乐观、悲观等不同情景，赋予相应概率，并使用时点参数（PIT）而非跨周期参数（TTC），将宏观经济预测（如 GDP、房价波动）直接纳入损失测算公式中。"
    },

    {
        "question": "在 PD-LGD 二维矩阵中，针对“高 PD / 低 LGD”的抵押依赖区资产，银行应采取什么样的管理策略？",
        "ground_truth": "针对这一象限，银行应采取“审慎准入”策略。虽然客户违约概率较高，但由于抵押品充足导致预期损失深度较低，管理核心应放在强化抵押品的实时监控与价值重估上。"
    },

    {
        "question": "什么是 RAROC？它在风控策略决策中起到了什么作用？",
        "ground_truth": "RAROC 指的是“风险调整后的资本收益率”，计算公式为：(收入 - 经营成本 - 预期损失) / 非预期损失（经济资本）。它的作用是将风险转化为财务语言，帮助风控官判断即便风险较高的资产，其利率定价是否足以覆盖 EL 并提供符合资本回报要求的收益。"
    },

    {
        "question": "风险偏好陈述（RAS）通常包含哪些具体的量化指标？",
        "ground_truth": "RAS 规定了银行愿意承担的最高风险水平，通常包括：资本底线指标（如资本充足率不低于 10.5%）、集中度限制指标（如单一行业贷款占比上限）以及资产质量目标指标（如不良贷款率 NPL 控制在 1.5% 以内）。"
    },

]


def run_evaluation():

    qwen_judge = ChatOpenAI(
        model="qwen-plus",
        api_key=SecretStr(os.getenv("QWEN_API_KEY") or ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )  #  type: ignore

    # 显式将裁判传给每个指标，防止内部跑偏去调 OpenAI
    m_faithfulness = Faithfulness(llm=qwen_judge)
    m_answer_relevancy = AnswerRelevancy(llm=qwen_judge)
    m_context_precision = ContextPrecision(llm=qwen_judge)
    m_context_recall = ContextRecall(llm=qwen_judge)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5") # 与 rag_pipeline.py 保持一致

    rag_chain = get_rag_chain()
    
    results = []
    
    for i, item in enumerate(test_questions):

        query = item["question"]
        
        try:           
        
            # 调用 RAG 链获取结果
            # 注意：这里需要获取检索到的原文
            response = rag_chain.invoke(
                {"input": query, "chat_history": []}
            )

            # 提取并清洗 Answer (确保是纯 str)
            raw_ans = response.get("answer", "")
            answer = raw_ans.content if hasattr(raw_ans, 'content') else str(raw_ans)

            # 提取并清洗 Contexts (确保是 List[str])
            raw_ctx = response.get("context", [])
            if not raw_ctx:
                contexts = ["未检索到相关背景知识"]
            else:
                contexts = [str(doc.page_content) for doc in raw_ctx]
        
            results.append({

                "user_input": query,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"]

            })

            print(f"RAG 生成进度: {i+1}/{len(test_questions)} - 已完成")

        except Exception as e:
            print(f"问题失败: {query}, 错误:{e}")
            traceback.print_exc()

    # 2. 转换为 RAGAS 所需的 Dataset 格式
    if not results:
        print("没有成功生成任何评测样本")
        return
    dataset = Dataset.from_list(results)

    print("\n[正在调用通义千问进行交叉评估，请耐心等待...]")

    try:
        # 3. 执行评估
        score: Any = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=qwen_judge,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=1, timeout=60)
        )

        print("\n--- 评估结果 ---")
        df = score.to_pandas()
        print(df)
    
        # 保存结果到 CSV 方便分析哪些案例失败了
        df.to_csv("rag_eval_results.csv", index=False, encoding="utf-8-sig")
    
    except Exception as e:
        print(f"评估中断，正在尝试挽救数据: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()
